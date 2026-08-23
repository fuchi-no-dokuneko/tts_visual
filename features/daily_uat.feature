@daily @real-model
Feature: Daily production TTS evaluator acceptance
  The evaluator must validate its environment, synthesize one bounded real sample,
  validate the resulting audio, and open the moved portable report.

  Scenario: Evaluate one sample with the configured production model
    Given real GPT-SoVITS model and reference inputs are configured
    When I run the evaluator preflight
    And I synthesize one bounded sample with the real model
    Then the run manifest and generated PCM audio are valid
    And Chromium opens the moved portable report and records a screenshot
