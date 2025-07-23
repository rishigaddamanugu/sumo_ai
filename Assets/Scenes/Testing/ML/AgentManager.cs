using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

public class AgentManager : MonoBehaviour
{
    public List<AgentUnit> agents = new List<AgentUnit>();
    private bool isWaitingForResponses = false;

    void Start()
    {
        StartCoroutine(ParallelAgentLoop());
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

    IEnumerator ProcessAllAgentsParallel()
    {
        if (agents.Count == 0) yield break;

        isWaitingForResponses = true;

        // Pause the game once for all agents
        Time.timeScale = 0f;

        // Start all agents making their API calls simultaneously
        var agentCoroutines = new List<Coroutine>();
        foreach (var agent in agents)
        {
            agentCoroutines.Add(StartCoroutine(agent.RequestAction()));
        }

        // Wait for ALL agents to complete their API calls
        foreach (var coroutine in agentCoroutines)
        {
            yield return coroutine;
        }

        // Resume game
        Time.timeScale = 1f;

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
}
