from gateway.redaction import TokenVault, redact, resolve_overlaps
from presidio_analyzer import RecognizerResult


def test_email_is_replaced_by_one_token_not_split_into_url_fragments():
    result, _ = redact("Reach me at john.smith@acme.com today.")

    assert "john.smith@acme.com" not in result.text
    assert "<EMAIL_ADDRESS_1>" in result.text
    assert "<URL" not in result.text


def test_structurally_valid_ssn_is_redacted():
    # 123-45-6789 is deliberately not used: Presidio rejects known-invalid SSNs,
    # so it would not be detected and the test would say nothing about redaction.
    result, _ = redact("Employee SSN: 536-90-4399.")

    assert "536-90-4399" not in result.text
    assert "<US_SSN_1>" in result.text


def test_same_value_gets_the_same_token_and_different_values_differ():
    result, _ = redact("Alice Johnson met Bob Brown. Later, Alice Johnson left.")

    assert result.text.count("<PERSON_1>") == 2
    assert "<PERSON_2>" in result.text


def test_vault_restores_the_original_text():
    original = "Contact Alice Johnson at alice.johnson@corp.com."
    result, vault = redact(original)

    assert vault.restore(result.text) == original


def test_text_without_pii_is_unchanged():
    text = "The Licensee shall pay all fees within thirty days of invoice."
    result, vault = redact(text)

    assert result.text == text
    assert vault.token_to_value == {}


def test_overlap_resolution_prefers_higher_score_then_longer_span():
    email = RecognizerResult("EMAIL_ADDRESS", 0, 19, 1.0)
    url = RecognizerResult("URL", 0, 7, 0.5)
    other = RecognizerResult("URL", 12, 19, 0.5)

    kept = resolve_overlaps([url, other, email])

    assert kept == [email]


def test_token_vault_is_consistent_across_calls():
    vault = TokenVault()
    assert vault.token_for("PERSON", "Ann") == vault.token_for("PERSON", "Ann")
    assert vault.token_for("PERSON", "Ann") != vault.token_for("PERSON", "Ben")
