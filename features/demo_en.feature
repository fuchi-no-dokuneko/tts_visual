@demo @english @real-model
Feature: English product introduction recording for tts_visual

  Scenario: Introduce a portable GPT-SoVITS comparison report
    Given a completed production comparison report is available
    And I begin a recorded demonstration
    When I open the comparison report for recording
    And I narrate in "en-US" for at least 10 seconds
      """
      tts_visual is a bounded GPT-SoVITS evaluator. It takes paired reference audio and text, synthesizes the same target prompt with selected model versions, and records each result in a reproducible manifest.
      """
    Then the recording view shows the report title and evaluated case count
    When I present the reference and generated audio columns
    And I narrate in "en-US" for at least 11 seconds
      """
      Each row keeps the subject text beside a reference player and one generated player per model version. Listen across the row to compare pronunciation, voice similarity, pacing, and artifacts under the same prompt.
      """
    When I present the portable and privacy behavior
    And I narrate in "en-US" for at least 10 seconds
      """
      The report uses relative local audio paths, so the complete folder can be moved and reviewed offline. Reference audio can also be omitted when a review bundle must not contain private source recordings.
      """
    Then I finish the recorded demonstration
