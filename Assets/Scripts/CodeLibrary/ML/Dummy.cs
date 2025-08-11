using UnityEngine;

public class Dummy : MonoBehaviour
{
    [Header("Ground Reference")]
    public GameObject ground;
    
    [Header("Death Detection")]
    [SerializeField] private float deathThreshold = 3f;
    
    private StadiumManager stadiumManager;
    
    void Start()
    {
        // Find the StadiumManager in the scene
        stadiumManager = FindObjectOfType<StadiumManager>();
        
        if (stadiumManager == null)
        {
            Debug.LogError("StadiumManager not found in the scene!");
        }
        
        if (ground == null)
        {
            Debug.LogWarning("Ground GameObject not assigned to Dummy!");
        }
    }
    
    void Update()
    {
        CheckDeathCondition();
    }
    
    private void CheckDeathCondition()
    {
        if (ground == null || stadiumManager == null) return;
        
        // Calculate the height difference between dummy and ground
        float heightDifference = transform.position.y - ground.transform.position.y;
        
        // Check if dummy has fallen below the threshold
        if (heightDifference <= deathThreshold)
        {
            // Call the StadiumManager's OnDummyDeath method
            stadiumManager.OnAgentDeath("bob");
            
            // Optional: Disable this script to prevent multiple calls
            enabled = false;
        }
    }
}
