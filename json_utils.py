"""
JSON handling and display utilities for the Gemini API search interface.
"""
import json
from typing import Dict, List, Any, Optional, Union


def flatten_deep_nested(obj: Any, current_key: str = "", level: int = 0) -> Union[Dict, List, str]:
    """
    Convert deeply nested structures (level 3+) into newline-separated strings.
    For level 3+, creates "key: value" format with newlines.
    """
    if level >= 2:  # Convert to newline-separated string for level 2+
        if isinstance(obj, dict):
            items = []
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    nested_key = f"{current_key}_{k}" if current_key else k
                    nested_result = flatten_deep_nested(v, nested_key, level + 1)
                    if isinstance(nested_result, str) and '\n' in nested_result:
                        items.append(nested_result)
                    else:
                        items.append(f"{nested_key}: {nested_result}")
                else:
                    items.append(f"{k}: {v}")
            return "\n".join(items)
        elif isinstance(obj, list):
            items = []
            for i, item in enumerate(obj):
                item_key = f"{current_key}_{i}" if current_key else str(i)
                if isinstance(item, (dict, list)):
                    nested_result = flatten_deep_nested(item, item_key, level + 1)
                    if isinstance(nested_result, str) and '\n' in nested_result:
                        items.append(nested_result)
                    else:
                        items.append(f"{item_key}: {nested_result}")
                else:
                    items.append(f"{item_key}: {item}")
            return "\n".join(items)
        else:
            return str(obj)
    else:
        if isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                result[k] = flatten_deep_nested(v, k, level + 1)
            return result
        elif isinstance(obj, list):
            return [flatten_deep_nested(item, current_key, level + 1) for item in obj]
        else:
            return obj


def limit_json_nesting_to_level2(json_data: Dict) -> Dict:
    """
    Limit JSON nesting to maximum level 2.
    Convert deeper nesting to newline-separated string format.
    """
    result = {}
    
    for key, value in json_data.items():
        if isinstance(value, dict):
            # Level 1 - keep as dict but process level 2
            level1_dict = {}
            for k2, v2 in value.items():
                if isinstance(v2, (dict, list)) and v2:  # Level 2+ - convert to newline-separated string
                    string_result = flatten_deep_nested(v2, k2, 2)
                    level1_dict[k2] = string_result
                else:
                    level1_dict[k2] = v2
            result[key] = level1_dict
        elif isinstance(value, list):
            # Level 1 - process list items
            processed_list = []
            for item in value:
                if isinstance(item, (dict, list)):
                    string_result = flatten_deep_nested(item, key, 2)
                    processed_list.append(string_result)
                else:
                    processed_list.append(item)
            result[key] = processed_list
        else:
            result[key] = value
    
    return result


def extract_json_from_response(response_text: str) -> tuple[Optional[Dict], str]:
    """Extract JSON from response text, handling prefix/suffix."""
    start = response_text.find('{')
    end = response_text.rfind('}') + 1
    
    if start != -1 and end > start:
        json_part = response_text[start:end]
        try:
            json_data = json.loads(json_part)
            return json_data, json_part
        except json.JSONDecodeError:
            return None, json_part
    return None, response_text


def json_to_markdown(json_data: Dict, level: int = 1) -> str:
    """Convert JSON to markdown with heading levels for indentation."""
    markdown = ""
    
    for key, value in json_data.items():
        heading = "#" * (level + 2)
        markdown += f"\n{heading} {key}\n\n"
        
        if isinstance(value, dict):
            markdown += json_to_markdown(value, level + 1)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    markdown += f"\n{heading}# Item {i + 1}\n\n"
                    markdown += json_to_markdown(item, level + 2)
                else:
                    markdown += f"- {item}\n"
            markdown += "\n"
        else:
            markdown += f"{value}\n\n"
    
    return markdown


def extract_fields_by_name(json_data: Dict, field_name: str, current_path: str = "") -> List[Dict[str, str]]:
    """
    Recursively extract all fields with a specific name from JSON.
    Returns a list of dicts with 'path' and 'value' keys.
    """
    results = []
    
    def traverse(obj: Any, path: str):
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_path = f"{path}_{key}" if path else key
                
                if key.lower() == field_name.lower():
                    results.append({
                        'path': new_path,
                        'value': str(value) if not isinstance(value, str) else value
                    })
                
                # Continue traversing even if we found the field, in case there are nested structures
                if isinstance(value, (dict, list)):
                    traverse(value, new_path)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                new_path = f"{path}_{i}" if path else str(i)
                traverse(item, new_path)
    
    traverse(json_data, current_path)
    return results


def extract_reference_fields(json_data: Dict) -> List[Dict[str, str]]:
    """Extract all 'reference' fields from JSON with their paths."""
    return extract_fields_by_name(json_data, 'reference')


def extract_url_fields(json_data: Dict) -> List[Dict[str, str]]:
    """Extract all 'url' fields from JSON with their paths."""
    return extract_fields_by_name(json_data, 'url') 