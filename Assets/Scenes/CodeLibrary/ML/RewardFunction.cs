using UnityEngine;

public class RewardFunction : MonoBehaviour
{
    [Header("Reward Settings")]
    public GameObject opponent;
    public GameObject ground;
    
    private Vector3 previousAgentPosition;
    private Vector3 previousOpponentPosition;
    private float previousDistance;
    private bool isInitialized = false;
    
    void Start()
    {
        if (opponent == null)
        {
            Debug.LogError("Opponent GameObject not assigned to RewardFunction!");
            return;
        }
        
        Initialize();
    }
    
    private void Initialize()
    {
        previousAgentPosition = transform.position;
        previousOpponentPosition = opponent.transform.position;
        previousDistance = Vector3.Distance(previousAgentPosition, previousOpponentPosition);
        isInitialized = true;
    }
    
    // Calculate reward based on distance change
    public float CalculateReward()
    {
        if (opponent == null || !isInitialized) return 0f;
        
        float reward = 0f;
        // If agent has hit ground, return very negative reward
        if (ground != null)
        {
            float groundY = ground.transform.position.y;
            float agentY = transform.position.y;
            float agentHeightDistance = Mathf.Abs(agentY - groundY);
            
            // Debug logging for death detection
            if (agentHeightDistance < 2f) // Log when close to death threshold
            {
                Debug.Log($"[RewardFunction] Agent close to ground: Y={agentY}, GroundY={groundY}, Distance={agentHeightDistance}");
            }
            
            if (agentHeightDistance < 1.5f) // Match the threshold used in AgentUnit
            {
                Debug.Log($"[RewardFunction] AGENT DIED! Returning -100 reward. Y={agentY}, GroundY={groundY}, Distance={agentHeightDistance}");
                return -100f; // Very negative reward for falling off
            }
        }
        
        // If opponent has hit ground, return super positive reward
        if (opponent != null && ground != null)
        {
            float groundY = ground.transform.position.y;
            float opponentY = opponent.transform.position.y;
            float opponentHeightDistance = Mathf.Abs(opponentY - groundY);
            if (opponentHeightDistance < 1.5f) // Match the threshold used in AgentUnit
            {
                return 100f; // Super positive reward for opponent falling
            }
        }
        
        
        
        Vector3 currentAgentPosition = transform.position;
        Vector3 currentOpponentPosition = opponent.transform.position;
        float currentDistance = Vector3.Distance(currentAgentPosition, currentOpponentPosition);
        
        // // Calculate reward: positive when getting closer, negative when moving away
        reward = reward + (previousDistance - currentDistance);
        
        // // Update previous values for next calculation
        previousAgentPosition = currentAgentPosition;
        previousOpponentPosition = currentOpponentPosition;
        previousDistance = currentDistance;
        
        return reward;
    }
    
    // Get current distance without updating previous values
    public float GetCurrentDistance()
    {
        if (opponent == null) return 0f;
        return Vector3.Distance(transform.position, opponent.transform.position);
    }
}
