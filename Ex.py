"""
Create a PromptLayer subfolder (named after full patient name) under a parent folder,
then publish the three vignette-generated patient build files into that subfolder.

Folder API (public v2):
  - List children:  GET https://api.promptlayer.com/api/public/v2/folders?parent_id=<id>
  - Create folder:  POST https://api.promptlayer.com/api/public/v2/folders

Publish templates (folder-targeted legacy endpoint):
  - POST https://api.promptlayer.com/prompt-templates
    Body: { folder_id, prompt_name, prompt_template: { type: "chat", messages: [...] }, commit_message?, metadata? }

Usage:
  python Ex.py \
    --parent-id 49463 \
    --patient "Trevor Ramirez" \
    --short /abs/path/patient_Trevor_short_build.md \
    --medium /abs/path/patient_Trevor_medium_build.md \
    --long /abs/path/patient_Trevor_long_build.md \
    --pretty --debug

Requires env var: PROMPTLAYER_API_KEY
"""

import os
import json
from typing import List, Dict, Optional, Any

import httpx


API_BASE = "https://api.promptlayer.com/api/public/v2"
API_BASE_LEGACY = "https://api.promptlayer.com"


def _headers(api_key: str) -> Dict[str, str]:
    return {
        "X-API-KEY": api_key,
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": "vignette-folder-publisher/1.0",
    }


def list_children(parent_id: int, api_key: str, debug: bool = False, workspace_id: Optional[int] = None) -> List[Dict]:
    headers = _headers(api_key)
    # Prefer legacy endpoints which allow listing children
    legacy_paths = [
        f"{API_BASE_LEGACY}/folders/{parent_id}/children",
        f"{API_BASE_LEGACY}/folders/{parent_id}",
    ]
    with httpx.Client(timeout=30.0) as client:
        for url in legacy_paths:
            r = client.get(url, headers=headers)
            if r.status_code == 200:
                try:
                    data = r.json()
                except Exception:
                    data = None
                if data is None:
                    continue
                if isinstance(data, list):
                    return [x for x in data if isinstance(x, dict)]
                if isinstance(data, dict):
                    # Common wrappers
                    for key in ("children", "folders", "results"):
                        val = data.get(key)
                        if isinstance(val, list):
                            return [x for x in val if isinstance(x, dict)]
                    # Sometimes directly a folder object; no children
                    return []
            else:
                if debug:
                    print(f"[DEBUG] GET {url} -> {r.status_code}; body={r.text[:200]}")

        # Fallback to public v2 listing if available (workspace-scoped first if provided)
        if workspace_id is not None:
            url_v2_ws = f"{API_BASE}/workspaces/{workspace_id}/folders?parent_id={parent_id}"
            r2 = client.get(url_v2_ws, headers=headers)
            if r2.status_code == 200:
                try:
                    data2 = r2.json()
                except Exception:
                    data2 = []
                return data2 if isinstance(data2, list) else []
            if debug:
                print(f"[DEBUG] GET {url_v2_ws} -> {r2.status_code}; body={r2.text[:200]}")

        url_v2 = f"{API_BASE}/folders?parent_id={parent_id}"
        r2 = client.get(url_v2, headers=headers)
        if r2.status_code != 200:
            if debug:
                print(f"[DEBUG] GET {url_v2} -> {r2.status_code}; body={r2.text[:200]}")
            print(f"❌ Failed to list children for parent_id={parent_id}: {r2.status_code}")
            return []
        try:
            data2 = r2.json()
        except Exception:
            return []
        return data2 if isinstance(data2, list) else []


def create_folder(name: str, parent_id: int, api_key: str, debug: bool = False, workspace_id: Optional[int] = None) -> Dict:
    headers = _headers(api_key)
    payload = {"name": name, "parent_id": parent_id}
    # Try workspace-scoped public v2 first if provided
    if workspace_id is not None:
        url_ws = f"{API_BASE}/workspaces/{workspace_id}/folders"
        with httpx.Client(timeout=30.0) as client:
            r_ws = client.post(url_ws, headers=headers, json=payload)
            if debug:
                print(f"[DEBUG] POST {url_ws} -> {r_ws.status_code}; body={payload}")
            try:
                body_ws = r_ws.json()
            except Exception:
                body_ws = {"raw": r_ws.text}
        if r_ws.status_code in (200, 201):
            print(f"✅ Created folder '{name}' (parent {parent_id}, workspace {workspace_id})")
            return body_ws
        # fall through to global endpoint if workspace-scoped fails

    # Global public v2
    url = f"{API_BASE}/folders"
    with httpx.Client(timeout=30.0) as client2:
        r = client2.post(url, headers=headers, json=payload)
        if debug:
            print(f"[DEBUG] POST {url} -> {r.status_code}; body={payload}")
        try:
            body = r.json()
        except Exception:
            body = {"raw": r.text}
    if r.status_code not in (200, 201):
        print(f"❌ Create folder failed for '{name}': {r.status_code} - {body}")
    else:
        print(f"✅ Created folder '{name}' (parent {parent_id})")
    return body


def _extract_folder_id_from_response(resp: Any) -> Optional[int]:
    if isinstance(resp, dict):
        # Direct id fields
        for key in ("id", "folder_id"):
            if key in resp:
                try:
                    return int(resp[key])
                except Exception:
                    pass
        # Nested common wrappers
        for wrapper in ("folder", "data", "result"):
            nested = resp.get(wrapper)
            if isinstance(nested, dict):
                val = _extract_folder_id_from_response(nested)
                if val is not None:
                    return val
        # Sometimes API returns list under data
        for wrapper in ("data", "results"):
            nested_list = resp.get(wrapper)
            if isinstance(nested_list, list) and nested_list:
                return _extract_folder_id_from_response(nested_list[0])
    return None


def ensure_patient_folder(parent_id: int, patient_full_name: str, api_key: str, debug: bool = False, workspace_id: Optional[int] = None) -> Optional[int]:
    # Best-effort: listing children may not be allowed; ignore failures
    try:
        existing = list_children(parent_id, api_key, debug=debug, workspace_id=workspace_id)
    except Exception:
        existing = []

    for f in existing:
        if isinstance(f, dict) and f.get("name", "").strip().lower() == patient_full_name.strip().lower():
            fid = f.get("id") or f.get("folder_id")
            try:
                return int(fid)
            except Exception:
                break

    # Create if missing
    created = create_folder(patient_full_name, parent_id, api_key, debug=debug, workspace_id=workspace_id)
    fid = _extract_folder_id_from_response(created)
    if fid is not None:
        return fid
    # Last resort: try listing again and pick by name
    existing_after = list_children(parent_id, api_key, debug=debug, workspace_id=workspace_id)
    for f in existing_after:
        if isinstance(f, dict) and f.get("name", "").strip().lower() == patient_full_name.strip().lower():
            fid2 = f.get("id") or f.get("folder_id")
            try:
                return int(fid2)
            except Exception:
                return None
    return None


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _split_into_segments(text: str) -> List[str]:
    """Split text into 2-5 non-empty segments for PromptLayer validation.

    Preference order:
    1) Split on markdown section breaks ("\n---\n" or variants)
    2) Split on double newlines
    3) Chunk by length
    Ensures at least 2 segments.
    """
    import re
    segments: List[str] = []
    # Robust split on lines of dashes (---) with optional surrounding whitespace
    dash_split = re.split(r"\n\s*---+\s*\n", text)
    dash_parts = [p.strip() for p in dash_split if p and p.strip()]
    if len(dash_parts) >= 2:
        segments = dash_parts[:5]
    if len(segments) < 2:
        parts = [p.strip() for p in text.split("\n\n") if p.strip()]
        segments = parts[:5]
    if len(segments) < 2:
        # Chunk by ~1500 chars
        chunk_size = 1500
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i+chunk_size].strip()
            if chunk:
                segments.append(chunk)
            if len(segments) >= 5:
                break
    # Guarantee at least 2 non-empty segments
    if len(segments) == 1:
        head = segments[0]
        mid = len(head)//2 or 1
        segments = [head[:mid].strip(), head[mid:].strip() or "(continued)"]
    return segments[:5]


def _parse_markdown_segments(text: str) -> Optional[List[Dict[str, Any]]]:
    """Parse markdown sections ### System / ### User / ### Assistant into messages.
    Returns a list of { role, content: [ {type:'text', text} ... ] } or None if not found.
    """
    import re
    # Normalize line endings
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    # Regex to capture sections starting with ### <Role>
    pattern = r"(?ms)^###\s*(System|User|Assistant)\s*\n(.*?)(?=\n^###\s*(System|User|Assistant)\s*\n|\Z)"
    matches = re.findall(pattern, t)
    if not matches:
        return None
    role_map = {"System": "system", "User": "user", "Assistant": "assistant"}
    messages: List[Dict[str, Any]] = []
    for role_label, body, _ in matches:
        role = role_map.get(role_label, "system")
        body_text = body.strip()
        # Prefer splitting the System body into multiple content segments
        contents: List[Dict[str, Any]] = []
        if role == "system":
            # Split on dashed section breaks first, then double newlines, then chunk
            sys_segments = _split_into_segments(body_text)
            # Guarantee at least 2 segments for system message
            if len(sys_segments) == 1:
                head = sys_segments[0]
                mid = len(head)//2 or 1
                sys_segments = [head[:mid].strip(), head[mid:].strip() or "(continued)"]
            contents = [{"type": "text", "text": seg} for seg in sys_segments[:5]]
        else:
            # Single content block is fine for user/assistant
            contents = [{"type": "text", "text": body_text}] if body_text else [{"type": "text", "text": ""}]
        messages.append({
            "role": role,
            "template_format": "jinja2",
            "input_variables": (["provider_question"] if role == "user" else []),
            "content": contents,
        })
    return messages if messages else None


def _smoke_print_messages(messages: List[Dict[str, Any]]) -> None:
    try:
        counts = [(m.get("role"), len(m.get("content", []))) for m in messages]
        preview = []
        for m in messages:
            role = m.get("role")
            texts = [c.get("text", "")[:80] for c in m.get("content", [])[:2]]
            preview.append(f"{role}:{len(m.get('content', []))} -> {' | '.join(texts)}")
        print(f"[SMOKE] segments-per-role={counts}")
        for line in preview:
            print(f"[SMOKE] {line}")
    except Exception:
        pass


def publish_prompt_in_folder(folder_id: int, prompt_name: str, content_text: str, api_key: str, commit_message: Optional[str], metadata: Optional[Dict[str, Any]], debug: bool = False, force_legacy: bool = False, dry_run: bool = False) -> Dict:
    # Primary: public v2 endpoint with explicit folder targeting
    url_v2 = f"{API_BASE}/prompt-templates"
    headers = _headers(api_key)
    
    # Prefer explicit markdown sections if present
    parsed_messages = _parse_markdown_segments(content_text)
    if parsed_messages is None:
        # Fallback: Build multiple content segments and two messages to satisfy validation
        segments = _split_into_segments(content_text)
        system_contents = [{"type": "text", "text": s} for s in segments]
        user_contents = [
            {"type": "text", "text": "CURRENT PROVIDER QUESTION:"},
            {"type": "text", "text": "{{ provider_question }}"},
        ]
        assistant_contents = [
            {"type": "text", "text": "Understood. I will respond as the patient persona following these guidelines."},
        ]
        messages = [
            {"role": "system", "template_format": "jinja2", "input_variables": ["provider_question"], "content": system_contents},
            {"role": "user", "template_format": "jinja2", "input_variables": ["provider_question"], "content": user_contents},
            {"role": "assistant", "template_format": "jinja2", "input_variables": ["provider_question"], "content": assistant_contents},
        ]
    else:
        # Prefer only system + user for compatibility; drop assistant entirely
        cleaned = []
        for m in parsed_messages:
            role = m.get("role")
            if role == "assistant":
                continue
            cleaned.append(m)
        messages = cleaned

    # Ensure system has >= 2 segments before attempting publish
    for m in messages:
        if m.get("role") == "system" and len(m.get("content", [])) < 2:
            # Force additional segmentation
            sys_text = "\n\n".join([c.get("text", "") for c in m.get("content", [])])
            forced = _split_into_segments(sys_text)
            if len(forced) < 2 and sys_text:
                mid = len(sys_text)//2 or 1
                forced = [sys_text[:mid].strip(), sys_text[mid:].strip() or "(continued)"]
            m["content"] = [{"type": "text", "text": seg} for seg in forced[:5]]

    if debug:
        _smoke_print_messages(messages)

    body_v2 = {
        "folder_id": folder_id,
        "prompt_name": prompt_name,
        "prompt_template": {
            "type": "chat",
            "messages": messages,
        },
    }
    if commit_message:
        body_v2["commit_message"] = commit_message
    if metadata:
        body_v2["metadata"] = metadata

    if dry_run:
        print(f"[DRY-RUN] Would publish '{prompt_name}' to folder {folder_id}")
        _smoke_print_messages(messages)
        return {"dry_run": True, "folder_id": folder_id, "prompt_name": prompt_name}

    v2_attempted = False
    if not force_legacy:
        with httpx.Client(timeout=30.0) as client:
            r = client.post(url_v2, headers=headers, json=body_v2)
            v2_attempted = True
            if debug:
                msg_counts = [len(m.get("content", [])) for m in body_v2["prompt_template"]["messages"]]
                print(f"[DEBUG] POST {url_v2} -> {r.status_code}; messages={len(body_v2['prompt_template']['messages'])}; segments={msg_counts}")
                try:
                    previews = []
                    for m in body_v2["prompt_template"]["messages"]:
                        role = m.get("role")
                        texts = [c.get("text", "")[:60] for c in m.get("content", [])[:2]]
                        previews.append(f"{role}: {' | '.join(texts)}")
                    for p in previews:
                        print(f"[DEBUG] {p}")
                except Exception:
                    pass
                if r.status_code not in (200, 201):
                    print(f"[DEBUG] Response body: {r.text[:500]}")
            try:
                data = r.json()
            except Exception:
                data = {"raw": r.text}
            if r.status_code in (200, 201):
                print(f"✅ Published '{prompt_name}' to folder {folder_id}")
                return data
            # If 404 on v2, fall through to legacy automatically
            if r.status_code not in (404, 405):
                # For other errors, proceed to legacy as well
                pass

    # Prepare REST-friendly messages by collapsing multiple content segments into one
    rest_messages = []
    try:
        for m in messages:
            texts = [c.get("text", "") for c in m.get("content", [])]
            combined = "\n\n".join([t for t in texts if t is not None])
            nm = dict(m)
            nm["content"] = [{"type": "text", "text": combined}]
            if "template_format" not in nm:
                nm["template_format"] = "jinja2"
            if "input_variables" not in nm:
                nm["input_variables"] = []
            rest_messages.append(nm)
    except Exception:
        rest_messages = messages

    # Fallbacks against legacy endpoints using REST and simpler payloads expected by server
    headers_legacy = _headers(api_key)
    headers_legacy["Authorization"] = f"Bearer {api_key}"

    # Attempt 0: REST endpoint with prompt_template and prompt_version (matches provided schema)
    url_rest = f"{API_BASE_LEGACY}/rest/prompt-templates"
    rest_body = {
        "prompt_template": {
            "prompt_name": prompt_name,
            "folder_id": folder_id,
            "prompt_template": {
                "type": "chat",
                "messages": rest_messages,
                "input_variables": [],
                "dataset_examples": [],
            },
            **({"commit_message": commit_message} if commit_message else {}),
            **({"metadata": metadata} if metadata else {}),
        },
        "prompt_version": {
            "prompt_name": prompt_name,
            "folder_id": folder_id,
            "prompt_template": {
                "type": "chat",
                "messages": rest_messages,
                "input_variables": [],
                "dataset_examples": [],
            },
            **({"commit_message": commit_message} if commit_message else {}),
            **({"metadata": metadata} if metadata else {}),
        },
    }
    with httpx.Client(timeout=30.0) as client_rest:
        rR = client_rest.post(url_rest, headers=headers_legacy, json=rest_body)
        if debug:
            print(f"[DEBUG] POST {url_rest} -> {rR.status_code}")
            if rR.status_code not in (200, 201):
                print(f"[DEBUG] Response body: {rR.text[:500]}")
        try:
            dataR = rR.json()
        except Exception:
            dataR = {"raw": rR.text}
    if rR.status_code in (200, 201):
        print(f"✅ Published '{prompt_name}' to folder {folder_id}")
        return dataR

    # Attempt 1: Legacy /prompt-templates with {name, folder_id, messages}
    payload_chat = {
        "name": prompt_name,
        "folder_id": folder_id,
        "messages": messages,
    }
    url_legacy = f"{API_BASE_LEGACY}/prompt-templates"
    with httpx.Client(timeout=30.0) as client:
        r2 = client.post(url_legacy, headers=headers_legacy, json=payload_chat)
        if debug:
            msg_counts2 = [len(m.get("content", [])) for m in messages]
            print(f"[DEBUG] POST {url_legacy} -> {r2.status_code}; messages={len(messages)}; segments={msg_counts2}")
            if r2.status_code not in (200, 201):
                print(f"[DEBUG] Response body: {r2.text[:500]}")
        try:
            data2 = r2.json()
        except Exception:
            data2 = {"raw": r2.text}
    if r2.status_code in (200, 201):
        print(f"✅ Published '{prompt_name}' to folder {folder_id}")
        return data2

    # Attempt 2: Folder-scoped endpoint /folders/{folder_id}/prompt-templates
    url_folder_legacy = f"{API_BASE_LEGACY}/folders/{folder_id}/prompt-templates"
    payload_folder_chat = {
        "name": prompt_name,
        "messages": messages,
    }
    with httpx.Client(timeout=30.0) as client_f:
        rF = client_f.post(url_folder_legacy, headers=headers_legacy, json=payload_folder_chat)
        if debug:
            print(f"[DEBUG] POST {url_folder_legacy} -> {rF.status_code}")
            if rF.status_code not in (200, 201):
                print(f"[DEBUG] Response body: {rF.text[:500]}")
        try:
            dataF = rF.json()
        except Exception:
            dataF = {"raw": rF.text}
    if rF.status_code in (200, 201):
        print(f"✅ Published '{prompt_name}' to folder {folder_id}")
        return dataF

    # Attempt 3: Completion fallback (some backends validate on content segments differently)
    err_msg = None
    try:
        err_msg = (data2 or {}).get("msg") if isinstance(data2, dict) else None
    except Exception:
        err_msg = None

    # Build completion content with at least 3 segments
    system_text = ""; user_text = "{{ provider_question }}"; extra = "(continued)"
    try:
        for m in messages:
            role = m.get("role")
            txt = "\n\n".join([c.get("text", "") for c in m.get("content", [])]).strip()
            if role == "system" and txt:
                system_text = txt
            if role == "user" and txt:
                user_text = txt
    except Exception:
        pass

    completion_payload = {
        "name": prompt_name,
        "folder_id": folder_id,
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": system_text}]},
            {"role": "user", "content": [
                {"type": "text", "text": user_text},
                {"type": "text", "text": extra},
            ]},
        ],
    }
    with httpx.Client(timeout=30.0) as client_c:
        rC = client_c.post(url_legacy, headers=headers_legacy, json=completion_payload)
        try:
            dataC = rC.json()
        except Exception:
            dataC = {"raw": rC.text}
    if rC.status_code in (200, 201):
        print(f"✅ Published (completion fallback) '{prompt_name}' to folder {folder_id}")
        return dataC

    print(f"❌ Publish failed for '{prompt_name}': rest={rR.status_code}, legacy={r2.status_code}, folder={rF.status_code}, completion={rC.status_code}")
    return {"rest": dataR, "legacy": data2, "folder": dataF, "completion": dataC}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Create a patient folder and publish three patient build prompts into it")
    parser.add_argument("--parent-id", type=int, default=49463, help="Parent folder ID under which to create the patient folder (default: 49463)")
    parser.add_argument("--folder-id", type=int, default=None, help="If provided, publish directly into this folder id and skip discovery/create")
    parser.add_argument("--patient", type=str, required=True, help="Full patient name (folder name and prompt name prefix)")
    parser.add_argument("--workspace-id", type=int, default=None, help="Optional PromptLayer workspace id for folder creation")
    parser.add_argument("--short", type=str, required=True, help="Path to short patient build file")
    parser.add_argument("--medium", type=str, required=True, help="Path to medium patient build file")
    parser.add_argument("--long", type=str, required=True, help="Path to long patient build file")
    parser.add_argument("--short-name", type=str, default=None, help="Override prompt name for short (default: '<Patient> - Short')")
    parser.add_argument("--medium-name", type=str, default=None, help="Override prompt name for medium (default: '<Patient> - Medium')")
    parser.add_argument("--long-name", type=str, default=None, help="Override prompt name for long (default: '<Patient> - Long')")
    parser.add_argument("--commit", type=str, default="Initial import from vignette generator", help="Commit message for publish")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    parser.add_argument("--debug", action="store_true", help="Debug HTTP printing and smoke screens")
    parser.add_argument("--dry-run", action="store_true", help="Build and display messages without publishing")
    parser.add_argument("--force-legacy", action="store_true", help="Skip v2 and publish to legacy endpoint directly")
    args = parser.parse_args()

    api_key = os.getenv("PROMPTLAYER_API_KEY")
    if not api_key:
        print("❌ PROMPTLAYER_API_KEY not found in environment")
        raise SystemExit(1)

    # Ensure/create patient folder (or use provided)
    folder_id = args.folder_id
    if folder_id is None:
        folder_id = ensure_patient_folder(args.parent_id, args.patient, api_key, debug=args.debug, workspace_id=args.workspace_id)
    if folder_id is None:
        payload = {"error": "Failed to ensure patient folder", "parent_id": args.parent_id, "patient": args.patient}
        print(json.dumps(payload, ensure_ascii=False, indent=2) if args.pretty else json.dumps(payload, ensure_ascii=False))
        raise SystemExit(2)

    # Read files
    short_text = read_text(args.short)
    medium_text = read_text(args.medium)
    long_text = read_text(args.long)

    # Build prompt names
    short_name = args.short_name or f"{args.patient} - Short"
    medium_name = args.medium_name or f"{args.patient} - Medium"
    long_name = args.long_name or f"{args.patient} - Long"

    published = []
    for label, name, text in (("short", short_name, short_text), ("medium", medium_name, medium_text), ("long", long_name, long_text)):
        meta = {"source": "vignette-generator", "patient": args.patient, "length": label}
        resp = publish_prompt_in_folder(
            folder_id,
            name,
            text,
            api_key,
            args.commit,
            meta,
            debug=args.debug,
            force_legacy=args.force_legacy,
            dry_run=args.dry_run,
        )
        if not isinstance(resp, dict) or "id" not in resp:
            payload = {"error": "Publish failed", "label": label, "prompt_name": name, "folder_id": folder_id, "response": resp}
            print(json.dumps(payload, ensure_ascii=False, indent=2) if args.pretty else json.dumps(payload, ensure_ascii=False))
            raise SystemExit(3)
        published.append({"label": label, "prompt_name": name, "response": resp})

    # Create six additional empty prompts in the same patient folder
    empty_prompt_names = [
        f"{args.patient} - Negative5Plus",
        f"{args.patient} - Negative3to5",
        f"{args.patient} - Very Short",
        f"{args.patient} - Response Length",
        f"{args.patient} - Negativity Analysis",
        f"{args.patient} - Empathy Analysis",
    ]
    empty_content = (
        "### System\n\n"
        "(empty placeholder)\n\n"
        "---\n\n"
        "### User\n\n{{ provider_question }}\n"
    )
    extras = []
    for pname in empty_prompt_names:
        try:
            r = publish_prompt_in_folder(
                folder_id,
                pname,
                empty_content,
                api_key,
                args.commit,
                {"source": "vignette-generator", "patient": args.patient, "length": "placeholder"},
                debug=args.debug,
                force_legacy=args.force_legacy,
                dry_run=args.dry_run,
            )
            extras.append({"name": pname, "response": r})
        except Exception as e:
            extras.append({"name": pname, "error": str(e)})

    result_payload = {"patient_folder_id": folder_id, "published": published, "extras": extras}
    if args.pretty:
        print(json.dumps(result_payload, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(result_payload, ensure_ascii=False))


if __name__ == "__main__":
    main()


