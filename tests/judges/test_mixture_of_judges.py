import pytest
from judgeval.judges.mixture_of_judges import build_dynamic_mixture_prompt

def test_build_dynamic_mixture_prompt_validation():
    sample_responses = ["Response 1", "Response 2"]
    
    # Test invalid system prompt type
    with pytest.raises(TypeError, match="Custom system prompt must be a string"):
        build_dynamic_mixture_prompt(sample_responses, custom_system_prompt=123)
    
    # Test empty system prompt
    with pytest.raises(ValueError, match="Custom system prompt cannot be empty"):
        build_dynamic_mixture_prompt(sample_responses, custom_system_prompt="")
    
    # Test invalid conversation history format
    invalid_conversation = ["not a dict"]
    with pytest.raises(TypeError, match="Custom conversation history must be a list of dictionaries"):
        build_dynamic_mixture_prompt(sample_responses, custom_conversation_history=invalid_conversation)
    
    # Test missing required keys in conversation messages
    invalid_messages = [{"role": "user"}]  # missing 'content'
    with pytest.raises(ValueError, match="Each message must have 'role' and 'content' keys"):
        build_dynamic_mixture_prompt(sample_responses, custom_conversation_history=invalid_messages)
    
    # Test invalid types for role and content
    invalid_types = [{"role": 123, "content": "test"}]
    with pytest.raises(TypeError, match="Message role and content must be strings"):
        build_dynamic_mixture_prompt(sample_responses, custom_conversation_history=invalid_types)
    
    # Test invalid role value
    invalid_role = [{"role": "invalid_role", "content": "test"}]
    with pytest.raises(ValueError, match="Message role must be one of: 'system', 'user', 'assistant'"):
        build_dynamic_mixture_prompt(sample_responses, custom_conversation_history=invalid_role)

def test_build_dynamic_mixture_prompt_success():
    sample_responses = ["Response 1", "Response 2"]
    valid_custom_prompt = "You are a helpful judge synthesizing responses."
    valid_conversation = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"}
    ]

    # Test with only judge responses
    result1 = build_dynamic_mixture_prompt(sample_responses)
    assert isinstance(result1, list)
    assert len(result1) == 6  # Default conversation has 6 messages
    assert result1[-1]['content'].endswith("Response 1\n# Judge 2's response: #\nResponse 2\n## End of Judge Responses ##\nSynthesized response:\n")

    # Test with custom system prompt
    result2 = build_dynamic_mixture_prompt(sample_responses, custom_system_prompt=valid_custom_prompt)
    assert isinstance(result2, list)
    assert len(result2) == 6  # Same length as default
    assert result2[0]['content'].startswith(valid_custom_prompt)
    assert "**IMPORTANT**: IF THE JUDGE RESPONSES ARE IN JSON FORMAT" in result2[0]['content']

    # Test with custom conversation history
    result3 = build_dynamic_mixture_prompt(sample_responses, custom_conversation_history=valid_conversation)
    assert isinstance(result3, list)
    assert len(result3) == 4  # 3 original messages + 1 new message
    assert result3[0:3] == valid_conversation
    assert "## Start of Judge Responses ##" in result3[-1]['content']

    # Test with both custom prompt and conversation history
    # This tests that custom_conversation_history overrides custom_system_prompt
    result4 = build_dynamic_mixture_prompt(
        sample_responses,
        custom_system_prompt=valid_custom_prompt,
        custom_conversation_history=valid_conversation
    )
    assert isinstance(result4, list)
    assert len(result4) == 4  # 3 original messages + 1 new message
    assert result4[0:3] == valid_conversation
    assert valid_conversation[0]['content'] in result4[0]['content']  # Verify custom prompt is included
    assert "## Start of Judge Responses ##" in result4[-1]['content']
    
