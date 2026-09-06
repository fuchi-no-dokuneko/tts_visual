@daily @real-model
Feature: Daily production TTS evaluator acceptance
  The evaluator must reject unsafe configuration, synthesize one bounded real sample,
  preserve explicit output policies, and render a private, portable comparison report.

  Scenario: Validate safe configuration and reject an unsafe precision transition
    Given real GPT-SoVITS model and reference inputs are configured
    When I run the evaluator preflight
    Then preflight succeeds without loading the model
    When I request float16 CPU execution in preflight
    Then preflight rejects the unsafe precision with a clear diagnostic

  Scenario: Evaluate one bounded sample with the configured production model
    Given real GPT-SoVITS model and reference inputs are configured
    When I synthesize one bounded sample with the real model
    Then the run manifest and generated PCM audio are valid

  Scenario: Inspect all visible comparison report content
    Given a completed production comparison report is available
    When I inspect the report document
    Then the report shows its title, case count, model column, subject text, and audio controls
    And the report contains no absolute private input or output paths

  Scenario: Move and reload the portable report
    Given a completed production comparison report is available
    When I copy the portable report to a different directory
    Then Chromium opens and reloads the moved report with local audio available

  Scenario: Show explicit privacy and missing-result states
    Given a completed production comparison report is available
    When I render a report that omits private reference audio
    Then the report labels the reference as Omitted and keeps generated audio
    When I render a report with one missing model result
    Then the report labels that model result as Missing

  Scenario: Refuse accidental replacement of existing output
    Given a completed production comparison report is available
    When I rerun against that output with the fail policy
    Then the evaluator refuses replacement and preserves the completed manifest
