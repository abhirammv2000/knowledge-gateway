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


def test_luhn_accepts_valid_and_rejects_invalid_numbers():
    from gateway.redaction import luhn_valid

    assert luhn_valid("4111111111111111")
    assert luhn_valid("639050142244")  # 12-digit, valid, missed by Presidio's default
    assert not luhn_valid("639050142243")  # one digit off


def test_twelve_digit_card_the_default_recognizer_misses_is_redacted():
    from gateway.redaction import luhn_valid

    number = "639050142244"
    assert luhn_valid(number)
    result, _ = redact(f"Payment shall be made by card number {number} monthly.")

    assert number not in result.text


def test_digit_run_that_fails_luhn_is_left_alone_by_the_card_recognizer():
    from gateway.redaction import luhn_valid

    number = "639050142243"
    assert not luhn_valid(number)
    result, _ = redact(f"Invoice reference number {number} is due upon receipt.")

    assert "CREDIT_CARD" not in {e for _, _, e in result.spans}
