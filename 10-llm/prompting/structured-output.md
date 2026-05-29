# Structured Output

## Table of Contents

1. [Why Structured Output Matters](#why-structured-output-matters)
2. [JSON Mode](#json-mode)
3. [Function Calling and Tool Use](#function-calling-and-tool-use)
4. [Structured Output with Schema Enforcement](#structured-output-with-schema-enforcement)
5. [Constrained Decoding Mechanics](#constrained-decoding-mechanics)
6. [Grammar-Constrained Generation](#grammar-constrained-generation)
7. [Pydantic Output Parsing with Instructor](#pydantic-output-parsing-with-instructor)
8. [XML and Markdown Output](#xml-and-markdown-output)
9. [Output Validation and Retry](#output-validation-and-retry)
10. [Extraction Tasks](#extraction-tasks)
11. [Format-Specific Prompt Patterns](#format-specific-prompt-patterns)
12. [References](#references)

---

## Why Structured Output Matters

Free-text LLM output is difficult to integrate into software systems. Every downstream consumer — a database, an API, a rendering layer — needs predictable structure. If the model outputs a paragraph when you needed a JSON object, parsing fails silently or noisily, and the system breaks.

The problem is more subtle than "just ask for JSON." LLMs are token predictors that do not inherently understand schema validity. Left unconstrained, they will:
- Add prose commentary before or after the JSON object.
- Use the wrong key names ("customer_name" vs "customerName").
- Omit required fields.
- Produce structurally valid JSON with semantically invalid values.
- Truncate output in the middle of a complex structure if they are nearing their context budget.

Structured output is a spectrum, from fragile (prompt-only), to reliable (JSON mode), to guaranteed (constrained decoding):

```
Prompt-only → JSON mode → Function calling → Schema-constrained decoding
  ↑ flexible, fragile                           ↑ rigid, guaranteed
```

---

## JSON Mode

OpenAI, Anthropic, and other providers offer a "JSON mode" that constrains the model to produce syntactically valid JSON.

### OpenAI JSON Mode

```python
import openai

response = openai.chat.completions.create(
    model="gpt-4o",
    response_format={"type": "json_object"},  # Enables JSON mode
    messages=[
        {
            "role": "system",
            "content": (
                "Extract information from user messages. "
                "Always output a JSON object with keys: "
                "name (string), age (integer), city (string)."
            )
        },
        {
            "role": "user",
            "content": "I'm Sarah, 29, from Toronto."
        }
    ]
)

import json
data = json.loads(response.choices[0].message.content)
# {"name": "Sarah", "age": 29, "city": "Toronto"}
```

**Critical requirement**: When using JSON mode with OpenAI, the word "json" must appear somewhere in the system or user message — otherwise the API returns an error. The system prompt above satisfies this.

### Limitations of JSON Mode

- **Syntactic but not semantic validity**: The output will be parseable JSON but may not conform to your schema. Keys may be wrong, types may mismatch, required fields may be absent.
- **No schema enforcement**: JSON mode ensures `json.loads()` succeeds, not that the object matches your `TypedDict` or Pydantic model.
- **Still can hallucinate values**: The values within the JSON object can still be incorrect.

For guaranteed schema adherence, use Structured Output (schema-constrained) or function calling, covered below.

---

## Function Calling and Tool Use

Function calling (OpenAI) / tool use (Anthropic) is the canonical way to request structured outputs from LLMs. The model signals its intent to call a function by outputting a structured tool call, which your code then executes and returns results from.

### Schema Definition

Tools are described using a JSON Schema-like format:

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "extract_person_info",
            "description": "Extract structured information about a person from text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Full name of the person"
                    },
                    "age": {
                        "type": "integer",
                        "description": "Age in years"
                    },
                    "email": {
                        "type": "string",
                        "description": "Email address if mentioned"
                    },
                    "occupation": {
                        "type": "string",
                        "description": "Job title or occupation if mentioned"
                    }
                },
                "required": ["name"],
                "additionalProperties": False
            }
        }
    }
]
```

### Making a Tool Call

```python
import openai
import json

response = openai.chat.completions.create(
    model="gpt-4o",
    tools=tools,
    tool_choice={"type": "function", "function": {"name": "extract_person_info"}},
    messages=[
        {
            "role": "user",
            "content": "Dr. Emily Chen, 34, is a cardiologist at Mayo Clinic. Reach her at echen@mayo.edu"
        }
    ]
)

# Parse the tool call
tool_call = response.choices[0].message.tool_calls[0]
arguments = json.loads(tool_call.function.arguments)
# {"name": "Dr. Emily Chen", "age": 34, "email": "echen@mayo.edu", "occupation": "cardiologist"}
```

### Tool Use Flow (Multi-Turn)

When the model needs to call a tool to answer a question (not just extract structured output), the flow involves returning tool results to the model:

```python
messages = [{"role": "user", "content": "What is the weather in Paris right now?"}]

# Round 1: Model requests a tool call
response = openai.chat.completions.create(
    model="gpt-4o",
    tools=weather_tools,
    messages=messages
)

tool_call = response.choices[0].message.tool_calls[0]
# tool_call.function.name = "get_weather"
# tool_call.function.arguments = '{"location": "Paris", "unit": "celsius"}'

# Execute the tool
weather_data = get_weather(json.loads(tool_call.function.arguments))

# Round 2: Return tool result, model generates final answer
messages.extend([
    response.choices[0].message,   # assistant message with tool_calls
    {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": json.dumps(weather_data)
    }
])

final_response = openai.chat.completions.create(
    model="gpt-4o",
    tools=weather_tools,
    messages=messages
)
# "The current temperature in Paris is 18°C with partly cloudy skies."
```

### Parallel Tool Calls

OpenAI models support requesting multiple tool calls in a single response (when `tool_choice="auto"` or when the task naturally requires it):

```python
# The model may return:
# tool_calls = [
#   ToolCall(name="get_weather", args={"location": "Paris"}),
#   ToolCall(name="get_weather", args={"location": "London"}),
#   ToolCall(name="convert_currency", args={"amount": 100, "from": "EUR", "to": "GBP"})
# ]

for tool_call in response.choices[0].message.tool_calls:
    result = dispatch_tool(tool_call.function.name, tool_call.function.arguments)
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": json.dumps(result)
    })
```

### Anthropic Tool Use

Anthropic's API follows the same pattern with slightly different syntax:

```python
import anthropic

client = anthropic.Anthropic()

tools = [
    {
        "name": "extract_entities",
        "description": "Extract named entities from text",
        "input_schema": {
            "type": "object",
            "properties": {
                "persons": {"type": "array", "items": {"type": "string"}},
                "organizations": {"type": "array", "items": {"type": "string"}},
                "locations": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["persons", "organizations", "locations"]
        }
    }
]

response = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=1024,
    tools=tools,
    tool_choice={"type": "tool", "name": "extract_entities"},
    messages=[{
        "role": "user",
        "content": "Apple Inc. was founded by Steve Jobs and Steve Wozniak in Cupertino."
    }]
)

# Find the tool_use block
for block in response.content:
    if block.type == "tool_use":
        entities = block.input
        # {"persons": ["Steve Jobs", "Steve Wozniak"],
        #  "organizations": ["Apple Inc."],
        #  "locations": ["Cupertino"]}
```

---

## Structured Output with Schema Enforcement

OpenAI's Structured Output feature (introduced 2024) uses constrained decoding to guarantee that the output matches a provided JSON Schema exactly.

```python
from pydantic import BaseModel
from typing import Optional

class PersonInfo(BaseModel):
    name: str
    age: Optional[int] = None
    occupation: Optional[str] = None
    email: Optional[str] = None

response = openai.chat.completions.parse(
    model="gpt-4o-2024-08-06",  # requires structured output capable model
    messages=[
        {"role": "user", "content": "Dr. Emily Chen, 34, cardiologist. echen@mayo.edu"}
    ],
    response_format=PersonInfo  # Pydantic model, not just type hint
)

person = response.choices[0].message.parsed
# person is a PersonInfo instance — guaranteed to be valid
assert isinstance(person.name, str)
assert isinstance(person.age, int)
```

**How it differs from JSON mode**: Schema enforcement uses constrained decoding (see next section) — the model cannot generate tokens that would violate the schema. Field types, required/optional status, and enum constraints are all guaranteed.

---

## Constrained Decoding Mechanics

The key question: how can a model be guaranteed to produce output matching a schema?

### Token Masking

At each decoding step, the model produces a probability distribution over the vocabulary. Constrained decoding **masks** (sets to $-\infty$ log probability) all tokens that would make the current partial output invalid according to the target grammar.

```
Partial output: {"name": "
Current state:  we are inside a JSON string value
Valid next tokens: any unicode character that can appear in a JSON string
Masked tokens:  unescaped " (would close the string prematurely), 
                unescaped backslash (invalid JSON escape), etc.
```

The resulting distribution is renormalized over unmasked tokens, and sampling proceeds normally. The model never has the opportunity to generate an invalid token.

### Finite-State Automata

JSON schemas are converted into a finite-state automaton (FSA) where:
- States represent positions in the partial parse (e.g., "inside object, expecting key or `}`", "inside string value for key `name`").
- Transitions on tokens move between states.
- Only transitions to valid states are allowed.

For each state, the set of tokens that have valid transitions can be precomputed as a bitmask over the vocabulary. These masks are applied at each decode step.

**Limitation**: The automaton must be derived from the schema at schema definition time. Complex schemas (deep nesting, large enums) produce large FSAs and can slow down decoding. For most practical schemas (< 20 fields, enums of < 100 values), the overhead is negligible.

---

## Grammar-Constrained Generation

For local/open-source LLMs, constrained generation is available via libraries that implement FSA-based token masking.

### Outlines

```python
import outlines
from pydantic import BaseModel
from typing import Literal

class Sentiment(BaseModel):
    label: Literal["positive", "negative", "neutral"]
    confidence: float
    reasoning: str

model = outlines.models.transformers("mistralai/Mistral-7B-Instruct-v0.2")
generator = outlines.generate.json(model, Sentiment)

result = generator("Classify: 'The product exceeded all my expectations!'")
# Sentiment(label='positive', confidence=0.95, reasoning='...')
# Guaranteed Pydantic-valid — impossible to get invalid label
```

### llama.cpp GBNF Grammars

llama.cpp supports GBNF (GGML BNF), a formal grammar notation that constrains generation:

```
# GBNF grammar for a sentiment JSON response
root   ::= "{" ws "\"sentiment\":" ws sentiment "," ws "\"score\":" ws score "}"
sentiment ::= "\"positive\"" | "\"negative\"" | "\"neutral\""
score  ::= [0-9] "." [0-9]+
ws     ::= [ \t\n]*
```

This grammar is compiled into a token mask that is applied at each decode step, guaranteeing the output matches the grammar exactly.

---

## Pydantic Output Parsing with Instructor

**Instructor** (Jason Liu) is the standard Python library for LLM-to-Pydantic extraction with automatic retry on validation failure.

```python
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator
from typing import Optional
import re

client = instructor.from_openai(OpenAI())

class ExtractedEvent(BaseModel):
    event_name: str = Field(description="Name of the event")
    date: str = Field(description="Date in ISO 8601 format (YYYY-MM-DD)")
    location: str = Field(description="City and country")
    attendees: Optional[int] = Field(
        default=None, 
        description="Number of expected attendees if mentioned"
    )

    @field_validator("date")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        if not re.match(r"\d{4}-\d{2}-\d{2}", v):
            raise ValueError("Date must be in YYYY-MM-DD format")
        return v

event = client.chat.completions.create(
    model="gpt-4o",
    response_model=ExtractedEvent,
    max_retries=3,  # Retry if Pydantic validation fails
    messages=[{
        "role": "user",
        "content": "NeurIPS 2025 will be held in Vancouver, Canada on December 9th, 2025."
                   "About 15,000 researchers are expected to attend."
    }]
)

# event.event_name = "NeurIPS 2025"
# event.date = "2025-12-09"
# event.location = "Vancouver, Canada"
# event.attendees = 15000
```

**How Instructor retry works**: If the model returns JSON that fails Pydantic validation, Instructor automatically adds the validation error to the conversation and calls the model again:

```
User: [original message]
Assistant: {"date": "December 9, 2025", ...}  ← fails YYYY-MM-DD validator
User (auto-injected): Validation failed: date must be in YYYY-MM-DD format.
                      Please fix and respond with corrected JSON.
Assistant: {"date": "2025-12-09", ...}  ← passes
```

---

## XML and Markdown Output

Not every use case requires JSON. For human-readable structured output, XML and Markdown are simpler and more robust:

### XML Tags for Multi-Part Output

```
You are a document analyzer. For each document, provide:
<summary>A 2-sentence summary.</summary>
<key_points>
  <point>First key point</point>
  <point>Second key point</point>
</key_points>
<action_required>YES or NO</action_required>
```

Parsing:

```python
import xml.etree.ElementTree as ET
import re

def extract_xml_section(text: str, tag: str) -> str:
    match = re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1).strip() if match else ""

summary = extract_xml_section(response, "summary")
action = extract_xml_section(response, "action_required")
```

### Markdown for Structured Reports

Markdown output is ideal when the consumer is a human or a renderer:

```
Generate a competitive analysis in this exact format:

## Overview
[2-3 sentences]

## Strengths
- [strength 1]
- [strength 2]

## Weaknesses
- [weakness 1]
- [weakness 2]

## Verdict
**Rating**: X/10 — [one sentence reason]
```

Parsing markdown sections:

```python
import re

def parse_markdown_sections(text: str) -> dict[str, str]:
    sections = re.split(r"^##\s+", text, flags=re.MULTILINE)
    result = {}
    for section in sections[1:]:  # skip preamble
        lines = section.strip().split("\n")
        heading = lines[0].strip()
        content = "\n".join(lines[1:]).strip()
        result[heading] = content
    return result
```

---

## Output Validation and Retry

Even with JSON mode or function calling, output validation and retry logic is essential for production systems.

```python
import json
import openai
from typing import TypeVar, Type, Callable
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

def validated_completion(
    messages: list[dict],
    model_class: Type[T],
    model: str = "gpt-4o",
    max_retries: int = 3
) -> T:
    """Call LLM and validate output against a Pydantic model, with retry."""
    errors = []
    
    for attempt in range(max_retries):
        # Add error feedback on retry
        if errors:
            messages = messages + [{
                "role": "user",
                "content": (
                    f"Your previous response failed validation:\n{errors[-1]}\n"
                    f"Please fix the issue and respond with valid JSON only."
                )
            }]
        
        response = openai.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=messages
        )
        
        try:
            raw = json.loads(response.choices[0].message.content)
            return model_class.model_validate(raw)
        except (json.JSONDecodeError, ValidationError) as e:
            errors.append(str(e))
            messages = messages + [{
                "role": "assistant",
                "content": response.choices[0].message.content
            }]
    
    raise RuntimeError(f"Failed after {max_retries} retries. Last error: {errors[-1]}")
```

---

## Extraction Tasks

Structured output is most commonly needed for information extraction: pulling structured data from unstructured text.

### Named Entity Extraction

```python
from pydantic import BaseModel
from typing import Optional
import instructor
from openai import OpenAI

class MedicalEntities(BaseModel):
    medications: list[str] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)
    dosages: list[str] = Field(default_factory=list)
    
client = instructor.from_openai(OpenAI())

entities = client.chat.completions.create(
    model="gpt-4o",
    response_model=MedicalEntities,
    messages=[{
        "role": "system",
        "content": "Extract medical entities from clinical notes. Be precise."
    }, {
        "role": "user",
        "content": "Patient started on Metformin 500mg BID for T2DM. "
                   "Lisinopril 10mg daily for hypertension. No allergies."
    }]
)
# medications = ["Metformin", "Lisinopril"]
# conditions = ["T2DM", "hypertension"]
# dosages = ["500mg BID", "10mg daily"]
```

### Relation Extraction

```python
class Relation(BaseModel):
    subject: str
    predicate: str  # e.g., "founded", "acquired", "is_subsidiary_of"
    object: str
    confidence: float = Field(ge=0.0, le=1.0)

class RelationSet(BaseModel):
    relations: list[Relation]
```

### Table Extraction from Text

```
Extract the data from the following text into a JSON array of objects.
Each object represents one row. Keys are the column headers.

Text: "Sales for Q1: North America $2.3M (+12%), Europe $1.8M (+5%),
Asia $3.1M (+22%). Q2: North America $2.5M (+9%), Europe $1.9M (+6%),
Asia $3.4M (+10%)."

Output the JSON array:
```

Expected output:
```json
[
  {"quarter": "Q1", "region": "North America", "sales_M": 2.3, "growth_pct": 12},
  {"quarter": "Q1", "region": "Europe", "sales_M": 1.8, "growth_pct": 5},
  ...
]
```

---

## Format-Specific Prompt Patterns

Prompt design has a large effect on structured output reliability, even with JSON mode active.

### Pattern 1: Schema in System Prompt, Data in User Message

```python
system = """
You are a data extraction API. Extract structured data from user-provided text.

Always output valid JSON matching this schema exactly:
{
  "entities": [
    {
      "text": "<exact span from input>",
      "type": "<PERSON | ORG | LOCATION | DATE | PRODUCT>",
      "start_char": <integer>,
      "end_char": <integer>
    }
  ]
}

Rules:
- "text" must be the exact substring from the input (copy verbatim)
- "start_char" and "end_char" are character offsets (0-indexed)
- Only include entities you are highly confident about
- Output nothing except the JSON object
"""
```

### Pattern 2: Demonstrate the Schema with a Worked Example

For complex schemas, include a complete example input-output pair in the system prompt:

```
Example input: "Apple acquired Beats Electronics in 2014 for $3 billion."
Example output:
{
  "acquirer": "Apple",
  "target": "Beats Electronics",
  "year": 2014,
  "price_usd_billions": 3.0
}
```

### Pattern 3: Assistant Prefill to Force JSON Start

On Anthropic, use an assistant prefill to force JSON output:

```python
response = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=1024,
    system="Extract information as JSON.",
    messages=[
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": "{"}  # Prefill forces JSON start
    ]
)
# The model continues from "{", guaranteed to start a JSON object
full_response = "{" + response.content[0].text
data = json.loads(full_response)
```

---

## References

- OpenAI. (2024). "Structured Outputs." platform.openai.com/docs/guides/structured-outputs [Schema enforcement via constrained decoding]
- OpenAI. (2023). "Function Calling." platform.openai.com/docs/guides/function-calling [Tool use API]
- Anthropic. (2024). "Tool Use (Function Calling)." docs.anthropic.com/en/docs/tool-use
- Liu, J. (2023). "Instructor: Structured LLM Outputs." github.com/jxnl/instructor [Pydantic + LLM]
- Willard, B.T., Louf, R. (2023). "Efficient Guided Generation for Large Language Models." arXiv:2307.09702 [Outlines — constrained decoding]
- Guidance. microsoft/guidance. [Token healing and grammar-constrained generation]
- Liang, P. et al. (2022). "Holistic Evaluation of Language Models." HELM. [Evaluation of structured extraction]
