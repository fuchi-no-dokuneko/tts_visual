@demo @cantonese @real-model
Feature: tts_visual 粵語產品介紹錄影

  Scenario: 介紹可攜式 GPT-SoVITS 比較報告
    Given a completed production comparison report is available
    And I begin a recorded demonstration
    When I open the comparison report for recording
    And I narrate in "yue-HK" for at least 10 seconds
      """
      tts_visual 係一個有明確上限嘅 GPT-SoVITS 評估工具。佢會讀取配對好嘅參考聲音同文字，用指定模型版本合成同一段目標文字，再將每個結果寫入可重現嘅記錄。
      """
    Then the recording view shows the report title and evaluated case count
    When I present the reference and generated audio columns
    And I narrate in "yue-HK" for at least 11 seconds
      """
      每一行會將內容文字、參考聲音播放器，同每個模型版本嘅生成播放器放埋一齊。沿住同一行逐個聽，就可以比較發音、聲線相似度、節奏同雜音。
      """
    When I present the portable and privacy behavior
    And I narrate in "yue-HK" for at least 10 seconds
      """
      報告使用相對本機聲音路徑，所以成個資料夾搬去另一部電腦都可以離線檢查。如果評審檔案唔可以包含私人原聲，亦可以選擇省略參考聲音。
      """
    Then I finish the recorded demonstration
