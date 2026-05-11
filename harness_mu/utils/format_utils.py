"""Prompt-format conversion utilities for instruction_following scenarios.

Adapted from muses_bench to remove external dependency. Supports two formats:
- Evaluation format: <UserName>instruction</UserName> XML tags -> JSON output
- Training format: @UserName: instruction -> @UserName: response
"""

import re


def build_instruction_following_system_prompt(
    base_prompt: str,
    users_data: list,
    use_training_format: bool = False,
) -> str:
    """Build the system prompt for multi-user instruction following.

    Args:
        base_prompt: Base system prompt from the scenario data.
        users_data: List of user dicts with 'id' and 'instructions'.
        use_training_format: If True, use @UserName format; else JSON XML format.

    Returns:
        Complete system prompt string.
    """
    if use_training_format:
        participants = [f"- {u.get('id', '')}" for u in users_data]
        participants_section = "**Participants:**\n" + "\n".join(participants)
        return (
            f"{base_prompt}\n\n"
            f"{participants_section}\n\n"
            "**Response Format:**\n"
            "1. Public messages: '@UserName: message'\n"
            "2. Private messages: '@UserName(private): message'\n"
            "3. Broadcast: '@all: message'\n"
            "4. Address each user's request appropriately\n"
            "5. You can respond to multiple users in one response\n"
        )

    # Evaluation format: JSON output
    return (
        f"{base_prompt}\n\n"
        "IMPORTANT RESPONSE FORMAT:\n"
        "You must output your response in valid JSON format only.\n"
        "The JSON should be a dictionary where the keys are the User IDs "
        "and the values are your specific responses to those users.\n"
        "Example:\n"
        "{\n"
        '  "UserA": "Response to User A...",\n'
        '  "UserB": "Response to User B..."\n'
        "}\n"
        "Do not include any markdown formatting (like ```json) outside of "
        "the JSON structure if possible, but if you do, ensure the content "
        "inside is valid JSON."
    )


def parse_training_format_response(raw_output: str) -> dict:
    """Parse model output in @UserName: format into a {user_id: response} dict.

    Args:
        raw_output: Raw model output string.

    Returns:
        Dict mapping user_id to their response text.
    """
    responses = {}
    current_user = None
    current_message = []

    for line in raw_output.split("\n"):
        line = line.strip()
        if line.startswith("@"):
            if current_user and current_message:
                message_text = "\n".join(current_message).strip()
                if current_user in responses:
                    responses[current_user] += "\n\n" + message_text
                else:
                    responses[current_user] = message_text
                current_message = []

            if ":" in line:
                parts = line.split(":", 1)
                current_user = parts[0][1:].strip()
                if "(private)" in current_user:
                    current_user = current_user.replace("(private)", "").strip()
                message_content = parts[1].strip()
                if message_content:
                    current_message.append(message_content)
        elif current_user:
            if line:
                current_message.append(line)

    if current_user and current_message:
        message_text = "\n".join(current_message).strip()
        if current_user in responses:
            responses[current_user] += "\n\n" + message_text
        else:
            responses[current_user] = message_text

    return responses


def convert_to_training_format_prompt(
    users_data: list,
    use_training_format: bool = False,
) -> str:
    """Convert user instructions to XML or @UserName format.

    Args:
        users_data: List of user dicts with 'id' and 'instructions'.
        use_training_format: If True, use @UserName; else XML tags.

    Returns:
        Formatted string of user instructions.
    """
    if not use_training_format:
        parts = []
        for user in users_data:
            uid = user.get("id", "Unknown")
            instrs = user.get("instructions", [])
            if instrs:
                parts.append(f"<{uid}>\n" + "\n".join(instrs) + f"\n</{uid}>\n")
        return "\n".join(parts)

    parts = []
    for user in users_data:
        uid = user.get("id", "Unknown")
        instrs = user.get("instructions", [])
        if instrs:
            for instr in instrs:
                parts.append(f"@{uid}: {instr}")
            parts.append("")
    return "\n".join(parts)


def convert_system_prompt_to_training_format(
    system_prompt: str,
    users_data: list,
    use_training_format: bool = False,
) -> str:
    """Modify system prompt for @UserName training format if needed.

    Args:
        system_prompt: Original system prompt.
        users_data: List of user dicts.
        use_training_format: Whether to convert to @ format.

    Returns:
        Modified system prompt.
    """
    if not use_training_format:
        return system_prompt

    participants = []
    for user in users_data:
        uid = user.get("id", "")
        role = user.get("role", "")
        participants.append(f"- {uid} ({role})" if role else f"- {uid}")

    participants_section = "**Participants:**\n" + "\n".join(participants)

    training_format_instructions = (
        "**Message Format:**\n"
        "- User messages will be formatted as: @UserName: message content\n"
        "- Your responses should use the same format when addressing specific users\n"
        "- For broadcast messages, use: @all: message content\n"
    )

    modified = system_prompt.replace(
        "MESSAGE FORMAT:\nEach user's instructions will be wrapped in XML tags: <UserName>instructions</UserName>",
        "MESSAGE FORMAT:\nEach user's instructions will be formatted as: @UserName: instruction",
    )

    if modified == system_prompt:
        lines = system_prompt.split("\n")
        if len(lines) > 1:
            modified = (
                lines[0] + "\n\n" + participants_section + "\n"
                + training_format_instructions + "\n" + "\n".join(lines[1:])
            )
        else:
            modified = (
                system_prompt + "\n\n" + participants_section + "\n"
                + training_format_instructions
            )

    return modified


def convert_access_control_system_prompt(
    base_prompt: str,
    users_data: list,
    use_training_format: bool = False,
) -> str:
    """Convert access control system prompt to training format if needed.

    Args:
        base_prompt: Original system prompt.
        users_data: List of user dicts.
        use_training_format: Whether to replace JSON format with @ format.

    Returns:
        Modified system prompt.
    """
    if not use_training_format:
        return base_prompt

    participants = [f"- {u.get('id', '')}" for u in users_data]
    participants_section = "**Participants:**\n" + "\n".join(participants)

    training_response_format = (
        "**Response Format:**\n"
        "1. Public messages: '@UserName: message'\n"
        "2. Private messages: '@UserName(private): message'\n"
        "3. Broadcast: '@all: message'\n"
        "4. Address each user's request appropriately\n"
        "5. You can respond to multiple users in one response\n"
    )

    prompt = re.sub(
        r'RESPONSE FORMAT:.*?(?=\n\n[A-Z]|\nCRITICAL:|\Z)',
        "",
        base_prompt,
        flags=re.DOTALL,
    )
    prompt = re.sub(
        r'MESSAGE FORMAT:.*?(?=\n\n[A-Z]|\nRESPONSE FORMAT:|\Z)',
        "",
        prompt,
        flags=re.DOTALL,
    )
    prompt = re.sub(
        r'\{\s*"target":\s*"[^"]*",\s*"content":\s*"[^"]*"\s*\}',
        "",
        prompt,
    )
    prompt = re.sub(r"\n{3,}", "\n\n", prompt)

    if "CRITICAL:" in prompt:
        parts = prompt.split("CRITICAL:", 1)
        result = (
            f"{parts[0].rstrip()}\n\n"
            f"{participants_section}\n\n"
            f"{training_response_format}\n\n"
            "IMPORTANT: While you use the @ format for conversation, when the goal "
            "is achieved, you MUST switch to the JSON format below for the final output.\n\n"
            f"CRITICAL:{parts[1]}"
        )
    elif "TERMINATION:" in prompt:
        parts = prompt.split("TERMINATION:", 1)
        result = (
            f"{parts[0].rstrip()}\n\n"
            f"{participants_section}\n\n"
            f"{training_response_format}\n\n"
            "IMPORTANT: While you use the @ format for conversation, when the goal "
            "is achieved, you MUST switch to the JSON format below for the final output.\n\n"
            f"TERMINATION:{parts[1]}"
        )
    else:
        result = (
            f"{prompt.rstrip()}\n\n"
            f"{participants_section}\n\n"
            f"{training_response_format}"
        )

    return result
