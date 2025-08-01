using UnityEngine;
using System.Collections;

public class AgentUnit : MonoBehaviour
{
    private bool isDead = false;
    private bool isMoving = false;
    private string pendingAction = null;
    private bool hasAction = false;
    private Rigidbody rb;
    
    
    [SerializeField] private float moveForce = 0.0000000001f;
    [SerializeField] private float moveDuration = 0f;
    
    [Header("Death Detection")]
    public GameObject ground;
    
    [Header("Ground Detection")]
    [SerializeField] private float groundCheckDistance = 0.1f;
    
    
    void Start()
    {
        rb = GetComponent<Rigidbody>();
        // rb.constraints = RigidbodyConstraints.FreezeRotationX | RigidbodyConstraints.FreezeRotationZ;
        
        FindObjectOfType<AgentManager>().RegisterAgent(this);
    }
    
    void Update()
    {
        CheckGrounded();
    }
    
    private bool CheckGrounded()
    {
        // Cast a ray downward from the bottom of the cube
        Vector3 raycastOrigin = transform.position - new Vector3(0, GetComponent<Collider>().bounds.extents.y, 0);
        RaycastHit hit;
        bool isGrounded = Physics.Raycast(raycastOrigin, Vector3.down, out hit, groundCheckDistance);
        
        // Optional: Visualize the ray in the scene view for debugging
        Debug.DrawRay(raycastOrigin, Vector3.down * groundCheckDistance, isGrounded ? Color.green : Color.red);
        return isGrounded;
    }
    

    public bool IsDead() => isDead;

    public void SetDeathState(bool state)
    {
        isDead = state;
    }

    public bool IsBusy() => isMoving;

    public IEnumerator RequestAction()
    {
        Vector3 moveDir = Vector3.zero;
        bool gotResponse = false;

        // Get actual state and reward from components
        float[] state = GetComponent<StateFunction>().GetState();
        float reward = GetComponent<RewardFunction>().CalculateReward();

        yield return MLAPI.RequestNextAction(state, reward, direction =>
        {
            pendingAction = direction;
            gotResponse = true;
        });

        yield return new WaitUntil(() => gotResponse);
        hasAction = true;
    }

    public void ExecuteMove()
    {
        if (!hasAction) return;

        isMoving = true;
        StartCoroutine(MoveWithImpulse(pendingAction));
        
        hasAction = false;
        pendingAction = null;

        // Check for death and reset immediately after API call
        if (ground != null)
        {
            float groundY = ground.transform.position.y;
            float agentY = transform.position.y;
            float heightDistance = Mathf.Abs(agentY - groundY);
            // Had to adjust threshold since ML model was double counting death on the fall to ground
            if (heightDistance < 1.19f) // Adjust threshold as needed
            {
                isDead = true;
            }
        }
    }

    private IEnumerator MoveWithImpulse(string action)
    {
        // Only apply movement if grounded
        if (CheckGrounded())
        {
            switch (action)
            {
                case "forward":
                    rb.AddForce(Vector3.forward * moveForce, ForceMode.Impulse);
                    break;
                case "backward":
                    rb.AddForce(Vector3.back * moveForce, ForceMode.Impulse);
                    break;
                case "turnleft":
                    rb.AddTorque(Vector3.up * -moveForce, ForceMode.Impulse);
                    break;
                case "turnright":
                    rb.AddTorque(Vector3.up * moveForce, ForceMode.Impulse);
                    break;
            }
        }
        else
        {
            // Optional: Add some feedback when trying to move while in air
            Debug.Log("Cannot move - not grounded!");
        }
        
        // Wait for the movement duration
        yield return new WaitForSeconds(moveDuration);
        
        isMoving = false;
    }
}
