using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class StadiumManager : MonoBehaviour
{
    [Header("Game Objects")]
    public GameObject ted;
    public GameObject bob;
    
    [Header("Scoreboard")]
    public TextMeshProUGUI tedScoreText;
    public TextMeshProUGUI bobScoreText;
    public TextMeshProUGUI roundText;
    public TextMeshProUGUI timerText;
    public TextMeshProUGUI timeElapsedText;
    
    // Store original positions
    private Vector3 tedOriginalPosition;
    private Vector3 bobOriginalPosition;
    private Quaternion tedOriginalRotation;
    private Quaternion bobOriginalRotation;
    
    // Score tracking
    private int tedScore = 0;
    private int bobScore = 0;
    private int round = 0;
    private int timer = 30;
    private float timeElapsed = 0f;

    void Start()
    {
        // Store original positions and rotations when the scene starts
        if (ted != null)
        {
            tedOriginalPosition = ted.transform.position;
            tedOriginalRotation = ted.transform.rotation;
        }
        
        if (bob != null)
        {
            bobOriginalPosition = bob.transform.position;
            bobOriginalRotation = bob.transform.rotation;
        }
        
        // Initialize scoreboard
        UpdateScoreboard();
        UpdateTimerText(timer);
    }
    
    void Update()
    {
        // Update the running stopwatch
        timeElapsed += Time.deltaTime;
        
        // Countdown timer
        if (timer > 0)
        {
            timer -= (int)Time.deltaTime;
            if (timer <= 0) resetStadium();
        }
        
        // Update timer display every frame
        UpdateTimerText(timer);
        
        // Update time elapsed display
        UpdateTimeElapsedText(timeElapsed);
    }
    
    public void resetStadium()
    {
        // Reset Ted to original position and rotation
        if (ted != null)
        {
            ted.transform.position = tedOriginalPosition;
            ted.transform.rotation = tedOriginalRotation;
        }
        
        // Reset Bob to original position and rotation
        if (bob != null)
        {
            bob.transform.position = bobOriginalPosition;
            bob.transform.rotation = bobOriginalRotation;
        }

        // Reset timer
        resetTimer();
        IncrementRound();
        
    }
    
    // Score management functions
    public void AddScoreToTed(int points = 1)
    {
        tedScore += points;
        UpdateScoreboard();
    }
    
    public void AddScoreToBob(int points = 1)
    {
        bobScore += points;
        UpdateScoreboard();
    }
    
    public void ResetScores()
    {
        tedScore = 0;
        bobScore = 0;
        UpdateScoreboard();
    }
    
    private void UpdateScoreboard()
    {
        if (tedScoreText != null)
        {
            tedScoreText.text = "Ted: " + tedScore.ToString();
        }
        
        if (bobScoreText != null)
        {
            bobScoreText.text = "Bob: " + bobScore.ToString();
        }
    }

    public void UpdateRoundText(int round)
    {
        if (roundText != null)
        {
            roundText.text = "Round: " + round.ToString();
        }
    }

    public void UpdateTimerText(int timer)
    {
        if (timerText != null)
        {
            timerText.text = "Timer: " + "00:" + timer.ToString("D2");
        }
    }


    public void IncrementRound()
    {
        round++;
        UpdateRoundText(round);
    }

    public void resetTimer()
    {
        timer = 30;
        UpdateTimerText(timer);
    }
    
    public void UpdateTimeElapsedText(float timeElapsed)
    {
        if (timeElapsedText != null)
        {
            timeElapsedText.text = "Elapsed Time: " + timeElapsed.ToString("F1") + "s";
        }
    }
    
    public void OnAgentDeath(string agentName)
    {
        // Add score to surviving agent
        Debug.Log("Agent " + agentName + " has died");
        
        if (agentName.Contains("Ted") || agentName.Contains("ted"))
        {
            AddScoreToBob(1);
        }
        else if (agentName.Contains("Bob") || agentName.Contains("bob"))
        {
            AddScoreToTed(1);
        }
        
        // Reset timer
        resetTimer();
        IncrementRound();
        
        // Reset stadium positions
        resetStadium();
    }
    
    // Getter methods for scores
    public int GetTedScore() { return tedScore; }
    public int GetBobScore() { return bobScore; }
}
