using UnityEngine;

public class RewardFunction : MonoBehaviour
{
    [Header("Reward Settings")]
    public GameObject opponent;
    
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
        
        Vector3 currentAgentPosition = transform.position;
        Vector3 currentOpponentPosition = opponent.transform.position;
        float currentDistance = Vector3.Distance(currentAgentPosition, currentOpponentPosition);
        
        // Calculate reward: positive when getting closer, negative when moving away
        float reward = previousDistance - currentDistance;
        
        // Update previous values for next calculation
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
