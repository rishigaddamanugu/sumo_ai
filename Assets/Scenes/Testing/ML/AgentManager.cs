using UnityEngine;
using System.Collections;
using System.Collections.Generic;

public class AgentManager : MonoBehaviour
{
    public List<AgentUnit> agents = new List<AgentUnit>();

    void Start()
    {
        StartCoroutine(SequentialAgentLoop());
    }

    IEnumerator SequentialAgentLoop()
    {
        while (true)
        {
            foreach (var agent in agents)
            {
                if (!agent.IsBusy())
                {
                    yield return StartCoroutine(agent.RequestAndMove());
                }
            }

            yield return null; // Small delay between full passes
        }
    }

    public void RegisterAgent(AgentUnit unit)
    {
        agents.Add(unit);
    }
}
