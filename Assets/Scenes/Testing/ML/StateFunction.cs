using UnityEngine;

public class StateFunction : MonoBehaviour
{
    [Header("State Settings")]
    public GameObject opponent;
    
    // Returns a simple state vector with agent and opponent positions
    public float[] GetState()
    {
        if (opponent == null)
        {
            Debug.LogError("Opponent GameObject not assigned to StateFunction!");
            return new float[6] { 0f, 0f, 0f, 0f, 0f, 0f };
        }
        
        Vector3 agentPos = transform.position;
        Vector3 opponentPos = opponent.transform.position;
        
        // Return state as: [agentX, agentY, agentZ, opponentX, opponentY, opponentZ]
        return new float[6] 
        {
            agentPos.x, agentPos.y, agentPos.z,
            opponentPos.x, opponentPos.y, opponentPos.z
        };
    }
    
    // Alternative: Get state as Vector3s if you prefer
    public Vector3 GetAgentPosition()
    {
        return transform.position;
    }
    
    public Vector3 GetOpponentPosition()
    {
        if (opponent == null) return Vector3.zero;
        return opponent.transform.position;
    }
}
