import streamlit as st
import json
import pandas as pd

st.title("Testing st.dataframe with JSON at Different Nesting Levels")

st.header("JSON Data Examples")

# Depth 1: Simple flat JSON
st.subheader("Depth 1 - Flat JSON")
json_depth_1 = {
    "name": "John Doe",
    "age": 30,
    "city": "New York",
    "salary": 75000,
    "active": True
}

st.code(json.dumps(json_depth_1, indent=2), language="json")
st.write("**Result with st.dataframe:**")
try:
    st.dataframe(json_depth_1)
except Exception as e:
    st.error(f"Error: {e}")

# Depth 2: Nested JSON with objects
st.subheader("Depth 2 - Nested JSON")
json_depth_2 = {
    "employee_id": 1,
    "personal_info": {
        "name": "Jane Smith",
        "age": 28,
        "email": "jane@example.com"
    },
    "work_info": {
        "department": "Engineering",
        "position": "Software Developer",
        "salary": {"value": 85000, "currency": "USD"} 
    },
    "active": True
}

st.code(json.dumps(json_depth_2, indent=2), language="json")
st.write("**Result with st.dataframe:**")
try:
    st.dataframe(json_depth_2)
except Exception as e:
    st.error(f"Error: {e}")

# Depth 3: Deeply nested JSON
st.subheader("Depth 3 - Deeply Nested JSON")
json_depth_3 = {
    "company": "Tech Corp",
    "employee": {
        "id": 123,
        "personal": {
            "name": "Bob Johnson",
            "contact": {
                "email": "bob@example.com",
                "phone": "555-0123",
                "address": {
                    "street": "123 Main St",
                    "city": "Boston",
                    "state": "MA"
                }
            }
        },
        "professional": {
            "department": "Data Science",
            "team": {
                "name": "ML Engineering",
                "manager": "Alice Wilson",
                "projects": {
                    "current": "AI Chatbot",
                    "completed": ["Data Pipeline", "Analytics Dashboard"]
                }
            }
        }
    }
}

st.code(json.dumps(json_depth_3, indent=2), language="json")
st.write("**Result with st.dataframe:**")
try:
    st.dataframe(json_depth_3)
except Exception as e:
    st.error(f"Error: {e}")

# Testing with list of JSON objects
st.subheader("List of JSON Objects (Different Depths)")

# List with depth 1 objects
st.write("**List of Depth 1 objects:**")
json_list_1 = [
    {"name": "Alice", "age": 25, "city": "Seattle"},
    {"name": "Bob", "age": 30, "city": "Portland"},
    {"name": "Charlie", "age": 35, "city": "Denver"}
]

st.code(json.dumps(json_list_1, indent=2), language="json")
try:
    st.dataframe(json_list_1)
except Exception as e:
    st.error(f"Error: {e}")

# List with depth 2 objects
st.write("**List of Depth 2 objects:**")
json_list_2 = [
    {
        "id": 1,
        "info": {"name": "Alice", "dept": "Engineering"},
        "metrics": {"score": 95, "projects": 3}
    },
    {
        "id": 2,
        "info": {"name": "Bob", "dept": "Marketing"},
        "metrics": {"score": 87, "projects": 5}
    }
]

st.code(json.dumps(json_list_2, indent=2), language="json")
try:
    st.dataframe(json_list_2)
except Exception as e:
    st.error(f"Error: {e}")

# Additional tests
st.header("Additional Tests")

st.subheader("Converting to DataFrame first")
st.write("**Converting nested JSON to DataFrame manually:**")
try:
    df_from_json = pd.json_normalize(json_depth_2)
    st.dataframe(df_from_json)
    st.write("✅ pd.json_normalize works for flattening nested JSON!")
except Exception as e:
    st.error(f"Error with pd.json_normalize: {e}")

st.subheader("Direct JSON string")
st.write("**Passing JSON as string:**")
json_string = json.dumps(json_depth_1)
st.code(json_string, language="json")
try:
    st.dataframe(json_string)
except Exception as e:
    st.error(f"Error: {e}")
