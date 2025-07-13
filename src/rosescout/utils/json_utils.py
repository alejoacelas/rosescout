"""JSON utilities for processing Gemini API responses."""
import json
from typing import Dict, List, Any, Optional, Tuple


def _flatten_deep_nested(obj: Any, level: int = 0) -> Any:
    """Convert deeply nested structures to strings at level 2+."""
    if level >= 2:
        if isinstance(obj, dict):
            items = [f"{k.upper()}: {v}" for k, v in obj.items()]
            return "\n\n".join(items)
        elif isinstance(obj, list):
            items = [f"{i}: {item}" for i, item in enumerate(obj)]
            return "\n\n".join(items)
        else:
            return str(obj)
    else:
        if isinstance(obj, dict):
            return {k: _flatten_deep_nested(v, level + 1) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_flatten_deep_nested(item, level + 1) for item in obj]
        else:
            return obj


def _remove_references(obj: Any) -> Any:
    """Remove 'references' fields recursively."""
    if isinstance(obj, dict):
        return {k: _remove_references(v) for k, v in obj.items() if k.lower() != 'references'}
    elif isinstance(obj, list):
        return [_remove_references(item) for item in obj]
    else:
        return obj


def _extract_references(obj: Any, path: str = "") -> List[Dict[str, str]]:
    """Extract all 'references' fields with their paths."""
    results = []
    
    def traverse(data: Any, current_path: str):
        if isinstance(data, dict):
            for key, value in data.items():
                new_path = f"{current_path}_{key}" if current_path else key
                
                if key.lower() == 'references':
                    if isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict):
                                result = item.copy()
                                result['path'] = new_path
                                results.append(result)
                    elif isinstance(value, dict):
                        result = value.copy()
                        result['path'] = new_path
                        results.append(result)
                
                if isinstance(value, (dict, list)):
                    traverse(value, new_path)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                traverse(item, f"{current_path}_{i}" if current_path else str(i))
    
    traverse(obj, path)
    return results


def extract_json_from_response(response_text: str) -> Tuple[Optional[Dict], str]:
    """Extract JSON from response text."""
    start = response_text.find('{')
    end = response_text.rfind('}') + 1
    
    if start != -1 and end > start:
        json_part = response_text[start:end]
        try:
            return json.loads(json_part), json_part
        except json.JSONDecodeError:
            return None, json_part
    return None, response_text


def extract_and_clean_json(json_data: Dict) -> Tuple[Dict, List[Dict[str, str]]]:
    """Extract references and return cleaned JSON."""
    references = _extract_references(json_data)
    cleaned = _remove_references(json_data)
    return cleaned, references


def limit_json_nesting_to_level2(json_data: Dict) -> Dict:
    """Limit JSON nesting to level 2, converting deeper levels to strings."""
    result = {}
    
    for key, value in json_data.items():
        if isinstance(value, dict):
            level1_dict = {}
            for k2, v2 in value.items():
                if isinstance(v2, (dict, list)) and v2:
                    level1_dict[k2] = _flatten_deep_nested(v2, 2)
                else:
                    level1_dict[k2] = v2
            result[key] = level1_dict
        elif isinstance(value, list):
            processed_list = []
            for item in value:
                if isinstance(item, (dict, list)):
                    processed_list.append(_flatten_deep_nested(item, 2))
                else:
                    processed_list.append(item)
            result[key] = processed_list
        else:
            result[key] = value
    
    return result

def convert_lists_to_strings(data):
    """Convert lists and complex objects in data to strings for dataframe compatibility."""
    if isinstance(data, list):
        return json.dumps(data, ensure_ascii=False, indent=2)
    elif isinstance(data, dict):
        return {k: convert_lists_to_strings(v) for k, v in data.items()}
    elif isinstance(data, (int, float, str, bool)) or data is None:
        return data
    else:
        # Convert any other objects to string representation
        return str(data)


def flatten_json_for_dataframe(data, parent_key='', sep='_'):
    """Flatten nested JSON structure for better dataframe display."""
    items = []
    if isinstance(data, dict):
        for k, v in data.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_json_for_dataframe(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                # Convert lists to string representation
                items.append((new_key, json.dumps(v, ensure_ascii=False, indent=2)))
            else:
                items.append((new_key, v))
    return dict(items)
