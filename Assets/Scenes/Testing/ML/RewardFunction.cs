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
            float agentDistance = Vector3.Distance(transform.position, ground.transform.position);
            if (agentDistance < 1f) // Adjust threshold as needed
            {
                return -100f; // Very negative reward for falling off
            }
        }
        
        // If opponent has hit ground, return super positive reward
        if (opponent != null && ground != null)
        {
            float opponentDistance = Vector3.Distance(opponent.transform.position, ground.transform.position);
            if (opponentDistance < 1f) // Adjust threshold as needed
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
