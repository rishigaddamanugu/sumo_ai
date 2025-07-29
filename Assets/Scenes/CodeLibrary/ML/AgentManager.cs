using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

public class AgentManager : MonoBehaviour
{
    public List<AgentUnit> agents = new List<AgentUnit>();
    private bool isWaitingForResponses = false;
    
    // Speed controller variables
    private float[] speedMultipliers = { 1f, 8f, 16f };
    private float[] speedDurations = { 5f, 5f, 15f };
    private int currentSpeedIndex = 0;
    private float speedTimer = 0f;
    private float previousTimeScale = 1f;
    private bool speedControllerActive = true;
    
    // Speed display UI reference
    private SpeedDisplayUI speedDisplayUI;

    void Start()
    {
        // Initialize speed display UI
        InitializeSpeedDisplay();
        
        StartCoroutine(ParallelAgentLoop());
        StartCoroutine(SpeedControllerLoop());
    }

    void InitializeSpeedDisplay()
    {
        // Create a GameObject with SpeedDisplayUI component
        GameObject speedDisplayGO = new GameObject("SpeedDisplayUI");
        speedDisplayUI = speedDisplayGO.AddComponent<SpeedDisplayUI>();
        
        // Parent it to this AgentManager for organization
        speedDisplayGO.transform.SetParent(transform);
        
        Debug.Log("SpeedDisplayUI initialized");
    }

    IEnumerator SpeedControllerLoop()
    {
        while (speedControllerActive)
        {
            // Update speed timer
            speedTimer += Time.unscaledDeltaTime;
            
            // Check if it's time to change speed
            if (speedTimer >= speedDurations[currentSpeedIndex])
            {
                speedTimer = 0f;
                currentSpeedIndex = (currentSpeedIndex + 1) % speedMultipliers.Length;
                
                // Only change time scale if we're not currently paused for API calls
                if (!isWaitingForResponses)
                {
                    float newSpeed = speedMultipliers[currentSpeedIndex];
                    Time.timeScale = newSpeed;
                    
                    // Update speed display
                    UpdateSpeedDisplay(newSpeed);
                }
            }
            
            yield return null;
        }
    }

    void UpdateSpeedDisplay(float speedMultiplier)
    {
        if (speedDisplayUI != null)
        {
            if (speedMultiplier > 1f)
            {
                speedDisplayUI.ShowSpeedText(speedMultiplier);
            }
            else
            {
                speedDisplayUI.HideSpeedText();
            }
        }
    }

    IEnumerator ParallelAgentLoop()
    {
        while (true)
        {
            // Wait until all agents are ready for their next action
            yield return new WaitUntil(() => agents.All(agent => !agent.IsBusy()));

            // Process all agents in parallel
            yield return StartCoroutine(ProcessAllAgentsParallel());

            yield return null; // Small delay between cycles
        }
    }

    public IEnumerator AgentDeathResetLoop(string agentName)
    {
        isWaitingForResponses = true;

        // Store current time scale before pausing
        previousTimeScale = Time.timeScale;
        
        // Pause the game
        Time.timeScale = 0f;

        StadiumManager stadiumManager = FindObjectOfType<StadiumManager>();
        if (stadiumManager != null)
        {
            stadiumManager.OnAgentDeath(agentName);
        }

        // Resume game with the previous time scale
        Time.timeScale = previousTimeScale;
        
        // Show speed display again if speed > 1x
        if (previousTimeScale > 1f)
        {
            UpdateSpeedDisplay(previousTimeScale);
        }

        isWaitingForResponses = false;
        
        yield return null; // Make it a proper coroutine
    }

    IEnumerator ProcessAllAgentsParallel()
    {
        if (agents.Count == 0) yield break;

        isWaitingForResponses = true;

        // Store current time scale before pausing
        previousTimeScale = Time.timeScale;
        
        // Pause the game once for all agents
        Time.timeScale = 0f;

        // Start all agents making their API calls simultaneously
        var agentCoroutines = new List<Coroutine>();
        foreach (var agent in agents)
        {
            agentCoroutines.Add(StartCoroutine(agent.RequestAction()));
        }
        bool anyAgentDied = false;
        foreach (var agent in agents)
        {
            if (agent.IsDead())
            {
                agent.SetDeathState(false);
                StartCoroutine(AgentDeathResetLoop(agent.name));
                anyAgentDied = true;
            }
        }

        if (anyAgentDied)
        {
            yield return null;
        }

        // Wait for ALL agents to complete their API calls
        foreach (var coroutine in agentCoroutines)
        {
            yield return coroutine;
        }

        // Resume game with the previous time scale
        Time.timeScale = previousTimeScale;
        
        // Show speed display again if speed > 1x
        if (previousTimeScale > 1f)
        {
            UpdateSpeedDisplay(previousTimeScale);
        }

        // Let all agents move simultaneously
        foreach (var agent in agents)
        {
            agent.ExecuteMove();
        }

        isWaitingForResponses = false;
    }

    public void RegisterAgent(AgentUnit unit)
    {
        agents.Add(unit);
    }

    public bool IsWaitingForResponses() => isWaitingForResponses;
    
    // Public methods to control speed controller
    public void SetSpeedControllerActive(bool active)
    {
        speedControllerActive = active;
        if (!active)
        {
            Time.timeScale = 1f; // Reset to normal speed when disabled
            if (speedDisplayUI != null)
            {
                speedDisplayUI.HideSpeedText();
            }
        }
    }
    
    public float GetCurrentSpeedMultiplier()
    {
        return speedMultipliers[currentSpeedIndex];
    }
    
    public float GetCurrentSpeedDuration()
    {
        return speedDurations[currentSpeedIndex];
    }
    
    public float GetRemainingTimeInCurrentSpeed()
    {
        return speedDurations[currentSpeedIndex] - speedTimer;
    }
    
    // Public methods to control speed display
    public void SetSpeedDisplayVisible(bool visible)
    {
        if (speedDisplayUI != null)
        {
            speedDisplayUI.SetSpeedTextVisible(visible);
        }
    }
    
    public void UpdateSpeedDisplaySettings(float fontSize, Color normalColor, Color speedupColor, Vector2 offset)
    {
        if (speedDisplayUI != null)
        {
            speedDisplayUI.UpdateUISettings(fontSize, normalColor, speedupColor, offset);
        }
    }
}
