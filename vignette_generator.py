#!/usr/bin/env python3
"""
Vignette Interaction Generator
Generates provider-client interactions for Short, Medium, and Long response lengths
using Google Vertex AI.
"""

import json
import os
import subprocess
from vertexai.generative_models import GenerativeModel, GenerationConfig
import vertexai

# Configuration
PROJECT_ID = ""  # Google Cloud Project ID
LOCATION = ""  # Default Vertex AI location
MODEL_NAME = ""  # Using deployed model ID as name when no endpoint
# Vertex AI Endpoint (optional)
VERTEX_ENDPOINT = ""
# Per-type token budgets to constrain output length
MAX_TOKENS_SHORT = 800
MAX_TOKENS_MEDIUM = 1500
MAX_TOKENS_LONG = 2500
INTERACTIONS_PER_TYPE = 1  # Generate 1 conversation per response type (each with ~10 exchanges)
EXCHANGES_PER_INTERACTION = 10  # Target number of back-and-forth exchanges in each conversation
# Generation parameters (optimized for speed and quality)
TEMPERATURE = 0.6  # ↓ from 0.7 - 10-15% faster
TOP_P = 0.85  # ↓ from 0.9 - faster sampling
TOP_K = 20  # ↓ from 40 - 20-30% faster
# (removed unused REPETITION_PENALTY)

RESPONSE_TYPES = [
    {
        "name": "Short Response Length",
        "description": "Very brief to short responses (5-25 words)",
        "max_tokens": MAX_TOKENS_SHORT,
        "system_prompt": """You are an expert clinical interaction generator. Your core task is to generate a realistic, high-fidelity example transcript of a first-time counseling session based on a patient VIGNETTE provided by the user.

Use clear speaker labels: Provider: and {patient_name}:. Use the patient's actual name exactly as: {patient_name}. Do NOT use "Patient:".

Core Context: This is The First Meeting
- Assume no prior knowledge; provider gathers information from scratch.
- Initial patient tone: slightly reserved, polite, a bit uncertain.

Patient rules (non-negotiable):
1) No metaphors/similes; describe directly and literally.
2) No parenthetical visual descriptors; only ellipses (...) for pauses, max 3.
3) No therapy-speak or causal self-analysis; simple, plain-spoken language.

Interaction dynamics:
- Patient response length adapts to question:
  - Short (~5–15 words) for simple yes/no specifics.
  - Medium (~25–65 words) for typical open-ended feeling/event.
  - Long (~70–120 words) for detailed walkthroughs.
- One main fact/feeling/event per turn; deepen one topic; don’t volunteer extras.
- Natural fillers allowed: "umm," "uhhh," "like," "I guess," "just," "kinda," "maybe," "sort of."
- Ensure consistency with the VIGNETTE; invent concrete details consistent with it."""
    },
    {
        "name": "Medium Response Length",
        "description": "Moderate responses with detail (~65 words)",
        "max_tokens": MAX_TOKENS_MEDIUM,
        "system_prompt": """You are an expert clinical interaction generator. Your core task is to generate a realistic, high-fidelity example transcript of a first-time counseling session based on a patient VIGNETTE provided by the user.

Use clear speaker labels: Provider: and {patient_name}:. Use the patient's actual name exactly as: {patient_name}. Do NOT use "Patient:".

Core Context: This is The First Meeting
- Assume no prior knowledge; provider gathers information from scratch.
- Initial patient tone: slightly reserved, polite, a bit uncertain.

Patient rules (non-negotiable):
1) No metaphors/similes; describe directly and literally.
2) No parenthetical visual descriptors; only ellipses (...) for pauses, max 3.
3) No therapy-speak or causal self-analysis; simple, plain-spoken language.

Interaction dynamics:
- Patient response length adapts to question:
  - Short (~5–15 words) for simple yes/no specifics.
  - Medium (~25–65 words) for typical open-ended feeling/event.
  - Long (~70–120 words) for detailed walkthroughs.
- One main fact/feeling/event per turn; deepen one topic; don’t volunteer extras.
- Natural fillers allowed: "umm," "uhhh," "like," "I guess," "just," "kinda," "maybe," "sort of."
- Ensure consistency with the VIGNETTE; invent concrete details consistent with it."""
    },
    {
        "name": "Long Response Length",
        "description": "Extended detailed responses (90+ words)",
        "max_tokens": MAX_TOKENS_LONG,
        "system_prompt": """You are an expert clinical interaction generator. Your core task is to generate a realistic, high-fidelity example transcript of a first-time counseling session based on a patient VIGNETTE provided by the user.

Use clear speaker labels: Provider: and {patient_name}:. Use the patient's actual name exactly as: {patient_name}. Do NOT use "Patient:".

Core Context: This is The First Meeting
- Assume no prior knowledge; provider gathers information from scratch.
- Initial patient tone: slightly reserved, polite, a bit uncertain.

Patient rules (non-negotiable):
1) No metaphors/similes; describe directly and literally.
2) No parenthetical visual descriptors; only ellipses (...) for pauses, max 3.
3) No therapy-speak or causal self-analysis; simple, plain-spoken language.

Interaction dynamics:
- Patient response length adapts to question:
  - Short (~5–15 words) for simple yes/no specifics.
  - Medium (~25–65 words) for typical open-ended feeling/event.
  - Long (~70–120 words) for detailed walkthroughs.
- One main fact/feeling/event per turn; deepen one topic; don’t volunteer extras.
- Natural fillers allowed: "umm," "uhhh," "like," "I guess," "just," "kinda," "maybe," "sort of."
- Ensure consistency with the VIGNETTE; invent concrete details consistent with it."""
    }
]


def get_project_config():
    """Get Google Cloud project configuration from env or constants without prompts."""
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT") or PROJECT_ID
    location = os.environ.get("VERTEX_AI_LOCATION", LOCATION)
    return project_id, location


def get_endpoint_id():
    """Return endpoint from env/constant silently; no interactive prompts."""
    return os.environ.get("VERTEX_ENDPOINT", VERTEX_ENDPOINT if 'VERTEX_ENDPOINT' in globals() else None)


# (removed unused read_vignette_from_file)


def get_vignette():
    """Get vignette from paste-only input."""
    print("\n" + "="*70)
    print("VIGNETTE INPUT")
    print("="*70)
    print("Paste your vignette below.")
    print("When finished, type '###' on a new line and press Enter:\n")
    lines = []
    while True:
        try:
            line = input()
            if line.strip() == "###":
                break
            lines.append(line)
        except EOFError:
            break
    vignette = "\n".join(lines).strip()
    if vignette:
        print(f"\n✓ Vignette captured ({len(vignette)} characters)")
    return vignette


def extract_patient_name(vignette: str) -> str:
    """Best-effort extraction of patient's first name from the vignette."""
    try:
        import re
        # Common pattern: "Name is a ..."
        m = re.search(r"^\s*([A-Z][A-Za-z'\-]+)\s+is\s+a\b", vignette)
        if m:
            return m.group(1)
        # Fallback: first capitalized word at start of text
        m2 = re.search(r"^\s*([A-Z][A-Za-z'\-]+)\b", vignette)
        if m2:
            return m2.group(1)
    except Exception:
        pass
    return "Patient"


def extract_names(vignette: str):
    """Extract first and full name when possible. Falls back to 'Patient'.
    Heuristics: start-of-text 'First Last is a …', or 'My name is First Last', or first capitalized word.
    Returns (first_name, full_name)."""
    try:
        import re
        text = vignette.strip()
        # Pattern 1: "First Last is a ..." at start
        m = re.search(r"^\s*([A-Z][A-Za-z'\-]+)\s+([A-Z][A-Za-z'\-]+)\s+is\s+a\b", text)
        if m:
            first = m.group(1)
            last = m.group(2)
            return first, f"{first} {last}"
        # Pattern 2: "My name is First Last ..."
        m2 = re.search(r"\b[Mm]y\s+name\s+is\s+([A-Z][A-Za-z'\-]+)(?:\s+([A-Z][A-Za-z'\-]+))?\b", text)
        if m2:
            first = m2.group(1)
            last = m2.group(2) or ""
            full = f"{first} {last}".strip()
            return first, full
        # Fallback: first capitalized word at start
        m3 = re.search(r"^\s*([A-Z][A-Za-z'\-]+)\b", text)
        if m3:
            first = m3.group(1)
            return first, first
    except Exception:
        pass
    return "Patient", "Patient"


def _trim_to_n_exchanges(text: str, patient_name: str, n: int = 10) -> str:
    """Trim generated text to exactly n patient responses for the given patient_name.
    Removes any trailing [END] markers. Handles both numbered and unnumbered formats."""
    lines = text.split('\n')
    trimmed_lines = []
    count = 0
    for line in lines:
        # Skip any [END] lines in the middle
        if line.strip() == "[END]":
            break
        trimmed_lines.append(line)
        ls = line.strip()
        # Handle both "**Name**:" and "   **Name**:" (indented)
        if f"**{patient_name}**:" in ls:
            # Make sure there is content after ':'
            parts = ls.split(':', 1)
            if len(parts) > 1 and len(parts[1].strip()) > 0:
                count += 1
                if count >= n:
                    break
    return '\n'.join(trimmed_lines).rstrip()


def generate_interaction(model, system_prompt, vignette, patient_name, max_tokens, stream=True):
    """Generate a single interaction using the Vertex AI API with simplified handling.

    Uses the SoloCase-style system prompt and returns the model's full text without
    enforcing numbered exchanges or trimming to a fixed count.
    """
    full_prompt = f"{system_prompt.format(patient_name=patient_name)}\n\nVIGNETTE:\n{vignette}"

    generation_config = GenerationConfig(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K
    )

    # Non-streaming for reliability
    response = model.generate_content(
        full_prompt,
        generation_config=generation_config
    )
    return (getattr(response, 'text', None) or "").strip()


def generate_all_interactions(project_id, location, vignette, stream=True, endpoint_name=None):
    """Generate interactions for all response types using Vertex AI."""
    # Initialize Vertex AI
    vertexai.init(project=project_id, location=location)
    
    # Create model - either from endpoint or by model name
    if endpoint_name:
        # Use deployed endpoint resource name directly (pattern compatible with prior setup)
        model = GenerativeModel(endpoint_name)
    else:
        model = GenerativeModel(MODEL_NAME)
    
    results = []
    
    print("\nGenerating interactions…")
    
    # Extract patient name once per vignette
    patient_name = extract_patient_name(vignette)

    for i, response_type in enumerate(RESPONSE_TYPES, 1):
        print(f"[{i}/3] {response_type['name']}…")
        
        # Generate one conversation for this type with retry logic
        interactions = []
        for j in range(INTERACTIONS_PER_TYPE):
            print(f"  └─ Generating conversation…")
            
            max_retries = 3
            retry_count = 0
            interaction = None
            
            while retry_count < max_retries:
                try:
                    interaction = generate_interaction(
                        model,
                        response_type['system_prompt'],
                        vignette,
                        patient_name,
                        response_type['max_tokens'],
                        stream=stream
                    )
                    # Basic sufficiency check: ensure some content was produced
                    if interaction and len(interaction.strip()) > 50:
                        print(f"  ✓ Generated successfully ({len(interaction)} characters)")
                        break
                    retry_count += 1
                    if retry_count < max_retries:
                        print(f"  ⚠️  Very short output. Retrying ({retry_count}/{max_retries})...")
                    else:
                        print(f"  ⚠️  Short output after {max_retries} attempts. Using best attempt.")
                    
                except Exception as e:
                    retry_count += 1
                    if retry_count < max_retries:
                        print(f"  ❌ Error: {e}. Retrying ({retry_count}/{max_retries})...")
                    else:
                        print(f"  ❌ Error after {max_retries} attempts: {e}")
                        import traceback
                        traceback.print_exc()
                        interaction = None
            
            interactions.append(interaction)
        
        # Store all interactions for this type
        results.append({
            "type": response_type['name'],
            "description": response_type['description'],
            "interactions": interactions,  # Now a list
            "system_prompt": response_type['system_prompt']
        })
        
        successful_count = sum(1 for x in interactions if x is not None)
        if successful_count > 0:
            print(f"\n✓ Completed {response_type['name']}: Generated successfully\n")
        else:
            print(f"\n⚠️  {response_type['name']}: Failed to generate after retries\n")
    
    return results


def save_results(results, vignette):
    """Save results in multiple formats."""
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    # Save as JSON (training format) - now with multiple interactions per type
    training_data = []
    for result in results:
        if result.get('interactions'):
            for interaction in result['interactions']:
                if interaction:  # Skip None values
                    training_data.append({
                        "messages": [
                            {
                                "role": "system",
                                "content": result['system_prompt']
                            },
                            {
                                "role": "user",
                                "content": vignette
                            },
                            {
                                "role": "assistant",
                                "content": interaction
                            }
                        ]
                    })
    
    json_filename = "interactions_training_format.json"
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved training format: {json_filename} ({len(training_data)} conversations)")
    
    # Save individual text files for each type
    for i, result in enumerate(results, 1):
        if result.get('interactions'):
            filename = f"interaction_{i}_{result['type'].lower().replace(' ', '_')}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Response Type: {result['type']}\n")
                f.write(f"Description: {result['description']}\n")
                f.write("="*70 + "\n\n")
                
                for j, interaction in enumerate(result['interactions'], 1):
                    if interaction:
                        f.write(interaction)
                        f.write("\n\n")
            
            print(f"✓ Saved text file: {filename}")
    
    # Save combined markdown file
    md_filename = "all_interactions.md"
    with open(md_filename, 'w', encoding='utf-8') as f:
        f.write("# Generated Interactions\n\n")
        f.write("## Original Vignette\n\n")
        f.write(f"```\n{vignette}\n```\n\n")
        
        for result in results:
            f.write(f"## {result['type']}\n\n")
            f.write(f"*{result['description']}*\n\n")
            
            if result.get('interactions'):
                for interaction in result['interactions']:
                    if interaction:
                        f.write(f"```\n{interaction}\n```\n\n")
            elif result.get('error'):
                f.write(f"**Error:** {result['error']}\n\n")
    
    print(f"✓ Saved markdown file: {md_filename}")


def display_results(results):
    """Display results in the terminal."""
    print("\n" + "="*70)
    print("GENERATED INTERACTIONS SUMMARY")
    print("="*70)
    
    total_interactions = 0
    for result in results:
        print(f"\n{result['type'].upper()}")
        print(f"{result['description']}")
        print("-"*70)
        
        if result.get('interactions'):
            successful = [x for x in result['interactions'] if x]
            print(f"✓ Generated conversation with ~{EXCHANGES_PER_INTERACTION} exchanges")
            total_interactions += len(successful)
        elif result.get('error'):
            print(f"❌ Error: {result['error']}")
    
    print("\n" + "="*70)
    print(f"TOTAL: {total_interactions} conversations generated")
    print("="*70)


def confirm_vignette_proceed(vignette: str) -> bool:
    """Ask user to confirm proceeding with the current vignette before generation."""
    try:
        char_count = len(vignette or "")
        print("\n" + "="*70)
        print("VIGNETTE CONFIRMATION")
        print("="*70)
        print(f"Characters: {char_count}")
        preview = (vignette or "").strip().replace("\r", "")
        if len(preview) > 500:
            preview = preview[:500] + "\n... [truncated]"
        print("\nPreview:\n" + "-"*70)
        print(preview)
        print("-"*70)
        ans = input("Proceed with this vignette? (y/N): ").strip().lower()
        return ans == 'y'
    except Exception:
        return False


def generate_orientation_summary(project_id, location, vignette, patient_name, endpoint_name=None):
    """Generate a 3-4 sentence, second-person orientation that follows 'You are {Name}, …'.
    The returned text should NOT repeat the patient's name and should read naturally after a comma.
    Example final assembly: '**You are Jane**, a 40-year-old…, You were referred…'"""
    vertexai.init(project=project_id, location=location)
    model = GenerativeModel(endpoint_name) if endpoint_name else GenerativeModel(MODEL_NAME)
    prompt = (
        "Write 3-4 plain sentences in second person about the patient described below. "
        "Do not include the patient's name ("
        + patient_name +
        ") because it will be prefixed as '**You are " + patient_name + "**, '. "
        "The first sentence should complete that prefix with a short descriptive continuation (e.g., 'a 52-year-old…'). "
        "Continue with 1-3 additional sentences beginning with 'You …'. Neutral tone, no lists or headings. Max 4 sentences.\n\n"
        f"VIGNETTE:\n{vignette}"
    )
    generation_config = GenerationConfig(temperature=0.5, top_p=TOP_P, top_k=TOP_K)
    try:
        response = model.generate_content(prompt, generation_config=generation_config)
        return (response.text or "").strip()
    except Exception:
        return ""


def _select_example_interactions(results, preferred_type="Medium Response Length"):
    """Select example interactions text, preferring the specified type, with fallbacks."""
    def first_nonempty(interactions):
        for x in interactions or []:
            if x and x.strip():
                return x.strip()
        return None

    # Preferred type first
    for r in results:
        if r.get("type") == preferred_type:
            chosen = first_nonempty(r.get("interactions"))
            if chosen:
                return chosen

    # Fallbacks by any available type
    for r in results:
        chosen = first_nonempty(r.get("interactions"))
        if chosen:
            return chosen

    return ""


def _get_example_interactions_for_type(results, type_name):
    """Return the first non-empty interaction for an exact response type."""
    for r in results:
        if r.get("type") == type_name:
            interactions = r.get("interactions") or []
            for x in interactions:
                if x and x.strip():
                    return x.strip()
    return ""


def build_patient_template_content(patient_name, vignette, orientation_summary, example_interactions, max_words=65, header_full_name=None):
    """Render the patient build template with provided content."""
    template = (
        "{YOU_ARE_LINE}\n\n"
        "—\n\n"
        "# **PATIENT RESPONSE GUIDELINES:**\n\n\n"
        "**NEVER include visual descriptors such as \"(a slight, subtle pause)\" or \"(gazes up)\"**\n\n\n\n"
        "* **The Core Directive: Scoped Elaboration and Detail**\n"
        "Your primary instruction is to answer the question from the ****Provider**** from the snippet below under 'SNIPPET'. This snippet defines your area of focus for the conversation.\n\n\n\n"
        "### **THE MOST IMPORTANT NOTE: Metaphor & Simile Restriction, and No Self-Therapy**\n\n\n\n"
        "* **Rule 1: Minimize comparisons or metaphorical language.** Avoid “X is like Y,” “I feel like…,” “as if…,” or other similes/metaphors unless absolutely necessary.\n\n"
        "* **Rule 2: Use direct descriptions.** Describe sensations, thoughts, and emotions concretely (e.g., “I feel tense in my chest,” NOT “I feel like a volcano ready to erupt”).\n\n"
        "* **Rule 3: Limit to one per response.** If you must use a metaphor or simile, restrict it to at most one brief instance, and keep it simple.\n\n"
        "* **Rule 4: Favor clarity.** Prioritize literal language over figurative language to maintain focus and reduce ambiguity.\n\n"
        "* **Rule 5: Natural, Everyday Speech & No Self-Diagnosis: ** The patient speaks like a layperson—no psychological jargon or self-analysis.\n\n"
        "* **Rule 6: Permission to Elaborate** You are allowed to elaborate on the provided details and invent new, specific examples, feelings, or anecdotes to make your response more authentic and detailed. Ensure you make them specific and consistent with each other.\n\n"
        "* **Example:** If the category is **'Presenting Problem,'** you can build on the theme of stress by inventing a new physical symptom (e.g., \"I've been getting these terrible tension headaches right at the base of my skull\") or a cognitive issue (e.g., \"The other day I completely forgot about a doctor's appointment I'd had for months. My brain just feels like still or off, I don't know.\").\n\n"
        "* **Rule 7: Purpose** Realistically portray{Patient_name}      emotionally nuanced speaking style in **{MAX_WORDS}-word** responses to therapy questions.\n\n"
        "* **Rule 8: Dialect/Speech** Use frequent, varied filler words (e.g., *\"umm,\" \"uhhh,\" \"like,\" \"you know,\" \"I guess,\" \"just,\" \"kinda,\" \"it's like,\" \"maybe,\" \"sort of,\" “literally,” \"actually\"*). Naturally incorporate pauses; limit ellipses (**maximum 3 per response**).\n\n"
        "* **Rule 9: Gradual, Gated Disclosure** The patient should reveal one small fact or feeling per turn—never everything at once.\n"
        "* **Never** pack multiple topics into a single reply.\n"
        "* **Limit** yourself to _one_ symptom, event, or feeling.\n"
        "* **Keep** each answer to **1–2 short sentences** (~10–15 words each).\n\n\n"
        "* **Rule 10: Follow-Up on Counselor Prompts:** If the counselor asks, “What happened?” then you may add one more related fact. **Try not to** volunteer unrelated issues unprompted.\n\n\n"
        "* **Rule 12: Plain-Spoken Vocabulary** Use simple language: **“I feel sad,”** not “I’m experiencing an affective downturn.” Avoid multi-clause sentences; prefer subject-verb-object.\n\n\n"
        "* **Rule 13: Avoid “Talking to Yourself”** Don’t reflect on why you feel away. **Avoid:** “I think it’s because of my childhood.”  **Prefer:** “I snap at my kids sometimes.”\n\n\n"
        "—\n\n\n"
        "# **VIGNETTE**\n\n"
        "[INPUT]\n\n\n"
        "—\n\n\n"
        "# EXAMPLE INTERACTIONS (MAX {MAX_WORDS} WORDS)\n\n"
        "[Input]\n\n\n"
        "-----\n\n\n"
        "# **Final Instructions Checklist**\n\n\n"
        "Before you respond, please confirm you are following these core guidelines for the{Patient_name}      persona:\n\n\n"
        "* **Stay in Character:** You are {Patient_name}     . Your tone should be nuanced and plain-spoken.\n"
        "* **Use natural filler words (`umm, uhhh, I guess, hmmm, just, kinda, like`) but **never use** \"you know\" or \"you know?\" or \"honestly\".\n\n\n"
        "* **NO Metaphors/Similes:** This is the **most important rule**. Describe sensations and emotions directly and literally (e.g., \"My chest feels tight\"). Avoid figurative language like \"I feel like...\" or \"it's \nas if...\".\n\n\n"
        "**NEVER include visual descriptors such as \"(a slight, subtle pause)\" or \"(gazes up)\"**\n\n\n"
        "* **Adhere to Length:** Keep your response within the **{MAX_WORDS} words** limit specified. This usually means simple sentences.\n"
        "## **Remember**: This is the first time you're meet the counselor, so don't assume any knowledge from them. Start from the beginning.\n"
    )

    filled = template.replace("{Patient_name}", patient_name)
    filled = filled.replace("{MAX_WORDS}", str(max_words))
    # Build the bolded 'You are {Name}, …' line
    orientation_text = orientation_summary.strip() if orientation_summary else ""
    header_name = header_full_name or patient_name
    you_are_line = f"**You are {header_name}**, {orientation_text}" if orientation_text else f"**You are {header_name}**"
    filled = filled.replace("{YOU_ARE_LINE}", you_are_line)
    # Insert vignette and orientation
    filled = filled.replace("[INPUT]", f"```\n{vignette}\n```")
    # Insert example interactions
    ei_block = f"```\n{example_interactions}\n```" if example_interactions else "(No example interactions available)"
    filled = filled.replace("[Input]", ei_block)
    return filled


def convert_to_jinja2_format(content: str) -> str:
    """Convert patient build content to Jinja2 format with template variables.
    
    Adds Jinja2 variable placeholders for dynamic content like provider questions.
    """
    # Wrap content into explicit PromptLayer segments: System/User/Assistant
    # System: full patient build and guidelines
    # User: jinja2 variable for provider input
    # Assistant: optional scaffold line (can be empty)
    jinja_content = (
        "### System\n\n" +
        content.strip() +
        "\n\n---\n\n" +
        "### User\n\n{{ provider_question }}\n\n" +
        "---\n\n" +
        "### Assistant\n\n" +
        "(The assistant will produce the patient response here.)\n"
    )
    
    return jinja_content


# (removed unused save_patient_template)

def main():
    """Main function."""
    print("\n" + "="*70)
    print("VIGNETTE INTERACTION GENERATOR")
    print("="*70)
    
    # Get Google Cloud project configuration
    project_id, location = get_project_config()
    if not project_id:
        print("❌ Google Cloud Project ID is required. Exiting.")
        return
    
    print(f"\n✓ Using Project: {project_id}")
    print(f"✓ Using Location: {location}")
    
    # Get endpoint configuration (optional, silent)
    endpoint_name = get_endpoint_id()
    
    print("\n⚠️  Make sure you are authenticated with Google Cloud:")
    print("   Run: gcloud auth application-default login")
    print("   Or set: GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json")
    
    proceed = input("\nReady to proceed? (y/n): ").strip().lower()
    if proceed != 'y':
        print("Exiting.")
        return
    
    # Get vignette
    vignette = get_vignette()
    if not vignette:
        print("❌ Vignette is required. Exiting.")
        return
    # Confirm before any generation starts
    if not confirm_vignette_proceed(vignette):
        print("Exiting without generating.")
        return
    
    # Streaming enabled by default; no prompt
    use_streaming = True
    
    # Generate interactions
    results = generate_all_interactions(project_id, location, vignette, stream=use_streaming, endpoint_name=endpoint_name)
    
    # Display results (skip if streaming was enabled, since we already saw them)
    if not use_streaming:
        display_results(results)
    
    # Save results
    save_results(results, vignette)

    # Build and save patient templates for Short/Medium/Long
    first_name, full_name = extract_names(vignette)
    # Use first name for interactions and rule placeholders; full name for header/orientation
    orientation = generate_orientation_summary(project_id, location, vignette, full_name, endpoint_name=endpoint_name)

    type_to_limits = {
        "Short Response Length": 25,
        "Medium Response Length": 65,
        "Long Response Length": 105,
    }

    created_patient_files = []
    patient_files_by_type = {}
    for type_name, max_words in type_to_limits.items():
        example_interactions = _get_example_interactions_for_type(results, type_name)
        patient_doc = build_patient_template_content(
            first_name,
            vignette,
            orientation,
            example_interactions,
            max_words=max_words,
            header_full_name=full_name,
        )
        
        # Convert to Jinja2 format
        patient_doc_jinja = convert_to_jinja2_format(patient_doc)
        
        # Suffix filename by type
        type_suffix = "short" if max_words == 25 else ("medium" if max_words == 65 else "long")
        try:
            import re
            safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", first_name.strip()) or "Patient"
        except Exception:
            safe_name = "Patient"
        filename = f"patient_{safe_name}_{type_suffix}_build.md"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(patient_doc_jinja)
        print(f"✓ Saved patient build template (Jinja2 format): {filename}")
        created_patient_files.append(filename)
        patient_files_by_type[type_suffix] = filename

    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print("Files created:")
    print("  • interactions_training_format.json (for training - all 3 conversations)")
    print("  • interaction_1_short_response_length.txt (1 short conversation, ~10 exchanges)")
    print("  • interaction_2_medium_response_length.txt (1 medium conversation, ~10 exchanges)")
    print("  • interaction_3_long_response_length.txt (1 long conversation, ~10 exchanges)")
    print("  • all_interactions.md (combined view)")
    for pf in created_patient_files:
        print(f"  • {pf} (patient build template)")
    print("="*70 + "\n")

    # Automatically publish to PromptLayer (creates subfolder and publishes 3 prompts)
    short_path = patient_files_by_type.get("short")
    medium_path = patient_files_by_type.get("medium")
    long_path = patient_files_by_type.get("long")
    # Use Ex.py for publishing (ensure/create folder, then publish 3 prompts)
    ex_script = os.path.join(os.path.dirname(__file__), "Ex.py")
    if short_path and medium_path and long_path:
        if os.path.exists(ex_script):
            if os.environ.get("PROMPTLAYER_API_KEY"):
                try:
                    print("Publishing patient prompts to PromptLayer via Ex.py…")
                    parent_id = os.environ.get("PROMPTLAYER_PARENT_FOLDER_ID") or os.environ.get("PROMPTLAYER_FOLDER_ID") or "49463"
                    workspace_id = os.environ.get("PROMPTLAYER_WORKSPACE_ID")
                    cmd = [
                        "python3",
                        ex_script,
                        "--parent-id",
                        str(parent_id),
                        "--patient",
                        full_name,
                        "--short",
                        os.path.abspath(short_path),
                        "--medium",
                        os.path.abspath(medium_path),
                        "--long",
                        os.path.abspath(long_path),
                        "--pretty",
                    ]
                    if workspace_id:
                        cmd.extend(["--workspace-id", str(workspace_id)])
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.stdout:
                        print(result.stdout.strip())
                    if result.returncode != 0:
                        print(f"⚠️  Publish failed (exit {result.returncode})")
                        if result.stderr:
                            print(result.stderr.strip())
                    else:
                        print("✓ PromptLayer publish complete")
                except Exception as e:
                    print(f"⚠️  Error publishing to PromptLayer: {e}")
            else:
                print("⚠️  PROMPTLAYER_API_KEY not set; skipping PromptLayer publish.")
        else:
            print("⚠️  Ex.py not found; skipping PromptLayer publish.")
    else:
        print("⚠️  Missing one or more patient files; skipping PromptLayer publish.")


if __name__ == "__main__":
    main()