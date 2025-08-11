using UnityEngine;
using System.Collections;

public class SceneController : MonoBehaviour
{
    public GameObject Agent;
    public GameObject Elevator;
    public GameObject SumoPlatform;
    public GameObject Stadium;
    public GameObject CubeCamera;
    public GameObject ElevatorCamera;
    public GameObject CubeSideCamera;
    public GameObject CubeRevolveCamera;
    public GameObject InbetweenCamera;
    public GameObject PlatformCamera;
    public GameObject Scoreboard;
    public GameObject ElevatorL;
    public GameObject ElevatorR;
    public GameObject TrapDoorL;
    public GameObject TrapDoorR;
    public GameObject CubeDistanceCamera;
    
    void Start()
    {
        StartCoroutine(ExecuteSceneBits());
    }
    
    private IEnumerator ExecuteSceneBits()
    {
        // yield return StartCoroutine(SceneBit1());
        yield return StartCoroutine(SceneBit2());
        yield return StartCoroutine(SceneBit3());
        yield return StartCoroutine(SceneBit4());
        yield return StartCoroutine(SceneBit5());
        yield return StartCoroutine(SceneBit6());
        yield return StartCoroutine(SceneBit7());
    }
    
    private IEnumerator SceneBit1()
    {
        SwitchToCubeCamera();
        yield return new WaitForSeconds(1f);
        // Cube is shown in elevator, and it is very clear by the shaking that elevator is moving
        Debug.Log("Scene Bit 1: Starting");
        
        // Start both elevator rumble and agent floating concurrently
        // StartCoroutine(AgentFloatAndFall());
        yield return StartCoroutine(RumbleElevator());
        yield return new WaitForSeconds(1f);
        
        Debug.Log("Scene Bit 1: Completed");
    }

    
    private IEnumerator SceneBit2()
    {
        // Elevator doors open (with a ding) and the cube slides out of the elevator
        Debug.Log("Scene Bit 2: Starting");
        
        // Switch to elevator camera
        SwitchToElevatorCamera();
        
        // Wait 1 second
        yield return new WaitForSeconds(2f);
        
        // Open elevator doors
        yield return StartCoroutine(OpenElevatorDoors());
        yield return StartCoroutine(CubeSlideToStadium());
        
        Debug.Log("Scene Bit 2: Completed");
    }

    private IEnumerator SceneBit3()
    {
        // Cube continues to slide to platform while elevator doors close concurrently
        // Have both coroutines execute at the same time
        Coroutine slideCoroutine = StartCoroutine(CubeSlideToSumoPlatform());
        Coroutine closeDoorsCoroutine = StartCoroutine(CloseElevatorDoors());
        
        // Wait for both coroutines to complete
        yield return slideCoroutine;
        yield return closeDoorsCoroutine;
    }

     private IEnumerator SceneBit4()
    {
        // Cube hops on platform
        Debug.Log("Scene Bit 4: Starting");
        SwitchToPlatformCamera();
        yield return StartCoroutine(CubeHopOnSumoPlatform());
        Debug.Log("Scene Bit 4: Completed");
    }
    

    private IEnumerator SceneBit5()
    {
        // Stadium rumbles, camera shakes, and platform rises
        Debug.Log("Scene Bit 5: Starting");
        yield return StartCoroutine(PlatformRiseSequence());
        Debug.Log("Scene Bit 5: Completed");
    }

    private IEnumerator SceneBit6()
    {
        // Open trap door
        Debug.Log("Scene Bit 6: Starting");
        yield return StartCoroutine(OpenTrapDoor());
        Debug.Log("Scene Bit 6: Completed");
    }

    private IEnumerator SceneBit7()
    {
        // Stadium and camera rumble, and scoreboard goes up
        Debug.Log("Scene Bit 4: Starting");
        yield return StartCoroutine(StadiumRumbleAndScoreboard());
        Debug.Log("Scene Bit 4: Completed");
    }

    private IEnumerator RumbleElevator()
    {
        Vector3 originalPosition = Elevator.transform.position;
        float rumbleDuration = 5f;
        float rumbleIntensity = 0.05f; // Gentle rumble intensity
        float rumbleSpeed = 20f; // How fast the rumble oscillates
        
        float elapsedTime = 0f;
        
        while (elapsedTime < rumbleDuration)
        {
            // Apply gentle random offset to elevator position
            float xOffset = Mathf.Sin(Time.time * rumbleSpeed) * rumbleIntensity;
            float yOffset = Mathf.Cos(Time.time * rumbleSpeed * 0.7f) * rumbleIntensity * 0.5f;
            float zOffset = Mathf.Sin(Time.time * rumbleSpeed * 1.3f) * rumbleIntensity * 0.3f;
            
            Elevator.transform.position = originalPosition + new Vector3(xOffset, yOffset, zOffset);
            
            elapsedTime += Time.deltaTime;
            yield return null;
        }
        
        // Return elevator to original position
        Elevator.transform.position = originalPosition;
    }
    private void SwitchToCubeCamera()
    {
        Debug.Log("Switching to Cube Camera");
        // Enable cube camera first, then disable others
        CubeCamera.SetActive(true);
        
        // Disable all other cameras
        ElevatorCamera.SetActive(false);
        CubeSideCamera.SetActive(false);
        CubeRevolveCamera.SetActive(false);
        InbetweenCamera.SetActive(false);
        PlatformCamera.SetActive(false);
        CubeDistanceCamera.SetActive(false);
    }

    private void SwitchToPlatformCamera()
    {
        Debug.Log("Switching to Platform Camera");
        PlatformCamera.SetActive(true);
        CubeCamera.SetActive(false);
        ElevatorCamera.SetActive(false);
        CubeSideCamera.SetActive(false);
        CubeRevolveCamera.SetActive(false);
        InbetweenCamera.SetActive(false);
        CubeDistanceCamera.SetActive(false);
    }

    private void SwitchToElevatorCamera()
    {
        Debug.Log("Switching to Elevator Camera");
        // Enable elevator camera first, then disable others
        ElevatorCamera.SetActive(true);
        
        // Disable all other cameras
        CubeCamera.SetActive(false);
        CubeSideCamera.SetActive(false);
        CubeRevolveCamera.SetActive(false);
        InbetweenCamera.SetActive(false);
        PlatformCamera.SetActive(false);
        CubeDistanceCamera.SetActive(false);
    }

    private void SwitchToInbetweenCamera()
    {
        Debug.Log("Switching to Inbetween Camera");
        // Enable inbetween camera first, then disable others
        InbetweenCamera.SetActive(true);
        ElevatorCamera.SetActive(false);
        CubeCamera.SetActive(false);
        CubeSideCamera.SetActive(false);
        CubeRevolveCamera.SetActive(false);
        PlatformCamera.SetActive(false);
        CubeDistanceCamera.SetActive(false);
    }

    private void SwitchToCubeDistanceCamera()
    {
        Debug.Log("Switching to Cube Distance Camera");
        CubeDistanceCamera.SetActive(true);
        ElevatorCamera.SetActive(false);
        CubeCamera.SetActive(false);
        CubeSideCamera.SetActive(false);
        CubeRevolveCamera.SetActive(false);
        PlatformCamera.SetActive(false);
        InbetweenCamera.SetActive(false);
    }
    private IEnumerator CubeHopOnSumoPlatform()
    {
        yield return new WaitForSeconds(1f);
        
        // Get the Rigidbody component from the Agent
        Rigidbody agentRigidbody = Agent.GetComponent<Rigidbody>();
        if (agentRigidbody == null)
        {
            Debug.LogError("Agent does not have a Rigidbody component!");
            yield break;
        }
        
        // Calculate platform-relative parameters
        Vector3 agentToPlatform = SumoPlatform.transform.position - Agent.transform.position;
        float distanceToPlatform = agentToPlatform.magnitude;
        
        // Calculate jump parameters based on distance to platform
        Vector3 jumpDirection = agentToPlatform.normalized; // Jump towards platform
        float jumpForce = Mathf.Max(8f, distanceToPlatform * 0.8f); // Scale upward force with distance
        float forwardForce = Mathf.Max(6f, distanceToPlatform * 0.6f); // Scale forward force with distance
        
        // Apply the jump force (upward + forward)
        Vector3 jumpVector = Vector3.up * jumpForce + jumpDirection * forwardForce;
        agentRigidbody.AddForce(jumpVector, ForceMode.Impulse);
        
        // Calculate jump duration based on distance
        float jumpDuration = Mathf.Max(3f, distanceToPlatform * 0.3f); // Scale duration with distance
        float rotateDuration = jumpDuration * 0.67f; // Rotation duration (slightly shorter than jump)
        float elapsedTime = 0f;
        
        Debug.Log($"Jumping and rotating simultaneously - Distance: {distanceToPlatform:F2}, Duration: {jumpDuration:F2}");
        Quaternion startRotation = Agent.transform.rotation;
        Quaternion endRotation = startRotation * Quaternion.Euler(0f, 90f, 0f);
        
        // Perform jump and rotation at the same time
        while (elapsedTime < jumpDuration)
        {
            // Handle rotation (only during the first part of the jump)
            if (elapsedTime < rotateDuration)
            {
                float rotationProgress = elapsedTime / rotateDuration;
                Agent.transform.rotation = Quaternion.Slerp(startRotation, endRotation, rotationProgress);
            }
            
            elapsedTime += Time.deltaTime;
            yield return null;
        }
        
        // Optional: Add a small downward force to ensure landing
        agentRigidbody.AddForce(Vector3.down * 2f, ForceMode.Impulse);
        
        Debug.Log("CubeHopOnSumoPlatform: Completed");
    }
    private IEnumerator OpenElevatorDoors()
    {
        Vector3 leftDoorOriginalPos = ElevatorL.transform.position;
        Vector3 rightDoorOriginalPos = ElevatorR.transform.position;
        
        float doorOpenDistance = 5f; // How far the doors move apart
        float doorOpenDuration = 1.5f; // How long the door opening takes
        
        float elapsedTime = 0f;
        
        while (elapsedTime < doorOpenDuration)
        {
            float progress = elapsedTime / doorOpenDuration;
            
            // Move left door to the left
            Vector3 leftDoorNewPos = leftDoorOriginalPos + Vector3.left * doorOpenDistance * progress;
            ElevatorL.transform.position = leftDoorNewPos;
            
            // Move right door to the right
            Vector3 rightDoorNewPos = rightDoorOriginalPos + Vector3.right * doorOpenDistance * progress;
            ElevatorR.transform.position = rightDoorNewPos;
            
            elapsedTime += Time.deltaTime;
            yield return null;
        }
        
        // Ensure doors are fully open
        ElevatorL.transform.position = leftDoorOriginalPos + Vector3.left * doorOpenDistance;
        ElevatorR.transform.position = rightDoorOriginalPos + Vector3.right * doorOpenDistance;
    }
    
    private IEnumerator CubeSlideToStadium()
    {
        // TODO: Implement cube sliding to platform and hopping on
        // Placeholder: wait 2 seconds
        yield return new WaitForSeconds(1f);
        // slide the cube to platform with constant speed
        // It basically just needs to move forward to the platform and then stop
        
        Vector3 startPosition = Agent.transform.position;
        float slideDistance = 10f; // How far to slide in -Z direction
        float slideDuration = 2f; // Slower slide
        float elapsedTime = 0f;
        
        while (elapsedTime < slideDuration)
        {
            float progress = elapsedTime / slideDuration;
            Vector3 newPosition = startPosition + Vector3.back * slideDistance * progress;
            Agent.transform.position = newPosition;
            
            // Switch to cube camera after 0.5 seconds
            if (elapsedTime >= 0.5f && elapsedTime < 0.6f)
            {
                SwitchToCubeCamera();
            }
            
            elapsedTime += Time.deltaTime;
            yield return null;
        }
        
        // Ensure final position
        Agent.transform.position = startPosition + Vector3.back * slideDistance;
    }
    
    private IEnumerator StadiumRumbleAndScoreboard()
    {
        // TODO: Implement stadium rumble and scoreboard rising
        // Placeholder: wait 2 seconds
        yield return new WaitForSeconds(2f);

        float targetY = 6f;
        float duration = 3f;
        Vector3 position = Scoreboard.transform.position;
        float startY = position.y;
        float distance = targetY - startY;
        float speed = distance / duration;

        float elapsedTime = 0f;

        while (elapsedTime < duration)
        {
            float moveThisFrame = speed * Time.deltaTime;
            position.y += moveThisFrame;
            Scoreboard.transform.position = position;

            elapsedTime += Time.deltaTime;
            yield return null;
        }

        // Stop exactly at target to avoid overshoot
        position.y = targetY;
        Scoreboard.transform.position = position;
    }
    
    private IEnumerator PlatformRiseSequence()
    {
        SwitchToPlatformCamera();

        // Parenting so the agent moves with the platform
        Agent.transform.SetParent(SumoPlatform.transform);

        yield return new WaitForSeconds(1f);

        float targetY = 10f;
        float duration = 4f;
        Vector3 position = SumoPlatform.transform.position;
        float startY = position.y;
        float distance = targetY - startY;
        float speed = distance / duration;

        float elapsedTime = 0f;

        while (elapsedTime < duration)
        {
            float moveThisFrame = speed * Time.deltaTime;
            position.y += moveThisFrame;
            SumoPlatform.transform.position = position;

            elapsedTime += Time.deltaTime;
            yield return null;
        }

        // Stop exactly at target to avoid overshoot
        position.y = targetY;
        SumoPlatform.transform.position = position;
        Agent.transform.SetParent(null);
    }

    
    private IEnumerator AgentFloatAndFall()
    {
        // wait one second
        yield return new WaitForSeconds(1f);

        Vector3 originalPosition = Agent.transform.position;
        float floatHeight = 5f; // How high the agent floats
        float floatDuration = 2.5f; // How long to float up
        float fallDuration = 2.5f; // How long to fall down
        float totalDuration = 5f; // Total duration to match elevator rumble
        
        float elapsedTime = 0f;
        
        while (elapsedTime < totalDuration)
        {
            float progress = elapsedTime / totalDuration;
            
            if (progress < 0.4f) // First half: float up
            {
                float floatProgress = progress * 2.5f; // 0 to 1 over first 40%
                float currentHeight = Mathf.Sin(floatProgress * Mathf.PI * 0.5f) * floatHeight; // Smooth float up
                Agent.transform.position = originalPosition + Vector3.up * currentHeight;
            }
            else if (progress < 0.5f) // Pause at top
            {
                // Stay at float height for 0.5 seconds
                Agent.transform.position = originalPosition + Vector3.up * floatHeight;
            }
            else // Second half: fall down
            {
                float fallProgress = (progress - 0.5f) * 2f; // 0 to 1 over second half
                float currentHeight = floatHeight - (fallProgress * (floatHeight - 1f)); // Fall from floatHeight to y=1
                Vector3 newPos = originalPosition;
                newPos.y = currentHeight;
                Agent.transform.position = newPos;
                
                if (newPos.y <= 2f)
                {
                    break;
                }
            }
            
            elapsedTime += Time.deltaTime;
            yield return null;
        }
    }

    private IEnumerator CubeSlideToSumoPlatform()
    {
        yield return new WaitForSeconds(2f);
        Vector3 startPosition = Agent.transform.position;
        float moveSpeed = 6f; // Units per second
        
        // Phase 1: Rotate 90 degrees clockwise, then move forward relative to cube
        float rotateDuration = 1f;
        float elapsedTime = 0f;
        
        Debug.Log("Phase 1: Rotating 90 degrees clockwise");
        Quaternion startRotation = Agent.transform.rotation;
        Quaternion endRotation = startRotation * Quaternion.Euler(0f, 90f, 0f);
        
        // Rotate first
        while (elapsedTime < rotateDuration)
        {
            float progress = elapsedTime / rotateDuration;
            Agent.transform.rotation = Quaternion.Slerp(startRotation, endRotation, progress);
            
            elapsedTime += Time.deltaTime;
            yield return null;
        }
        
        // Now move forward relative to the rotated cube
        float phase1Distance = 10f;
        float phase1Duration = phase1Distance / moveSpeed;
        elapsedTime = 0f;
        
        Debug.Log("Phase 1: Moving forward relative to cube");
        while (elapsedTime < phase1Duration)
        {
            float progress = elapsedTime / phase1Duration;
            Vector3 newPosition = startPosition + Agent.transform.forward * phase1Distance * progress;
            Agent.transform.position = newPosition;
            
            elapsedTime += Time.deltaTime;
            yield return null;
        }
        
        // Rotate 90 degrees counter-clockwise before Phase 2
        rotateDuration = 1f;
        elapsedTime = 0f;
        
        Debug.Log("Rotating 90 degrees counter-clockwise before Phase 2");
        startRotation = Agent.transform.rotation;
        endRotation = startRotation * Quaternion.Euler(0f, -90f, 0f);
        
        while (elapsedTime < rotateDuration)
        {
            float progress = elapsedTime / rotateDuration;
            Agent.transform.rotation = Quaternion.Slerp(startRotation, endRotation, progress);
            
            elapsedTime += Time.deltaTime;
            yield return null;
        }
        
        // Phase 2: Move -Z (forward)

        Vector3 phase2StartPos = Agent.transform.position;
        float phase2Distance = SumoPlatform.transform.position.z - Agent.transform.position.z;
        float phase2Duration = Mathf.Abs(phase2Distance) / moveSpeed;
        elapsedTime = 0f;
        
        Debug.Log("Phase 2: Moving forward (-Z)");
        while (elapsedTime < phase2Duration)
        {
            float progress = elapsedTime / phase2Duration;
            Vector3 newPosition = phase2StartPos + Vector3.forward * phase2Distance * progress;
            Agent.transform.position = newPosition;

            if (elapsedTime >= 0.5f && elapsedTime < 0.6f) 
            {
                SwitchToCubeDistanceCamera();
            }
            
            elapsedTime += Time.deltaTime;
            yield return null;
        }
        // Rotate 90 degrees counter-clockwise before Phase 2

        SwitchToPlatformCamera();
        rotateDuration = 1f;
        elapsedTime = 0f;
        
        Debug.Log("Rotating 90 degrees counter-clockwise before big jump.");
        startRotation = Agent.transform.rotation;
        endRotation = startRotation * Quaternion.Euler(0f, -90f, 0f);
        
        while (elapsedTime < rotateDuration)
        {
            float progress = elapsedTime / rotateDuration;
            Agent.transform.rotation = Quaternion.Slerp(startRotation, endRotation, progress);
            
            elapsedTime += Time.deltaTime;
            yield return null;
        }
        
        Debug.Log("CubeSlideToSumoPlatform: Completed");
    }
    
    private IEnumerator CloseElevatorDoors()
    {
        yield return new WaitForSeconds(1f);
        Vector3 leftDoorCurrentPos = ElevatorL.transform.position;
        Vector3 rightDoorCurrentPos = ElevatorR.transform.position;
        
        // Calculate the original positions (doors are currently open)
        Vector3 leftDoorOriginalPos = leftDoorCurrentPos + Vector3.right * 5f; // Move right to close
        Vector3 rightDoorOriginalPos = rightDoorCurrentPos + Vector3.left * 5f; // Move left to close
        
        float doorCloseDistance = 5f; // How far the doors move to close
        float doorCloseDuration = 1.5f; // How long the door closing takes
        
        float elapsedTime = 0f;
        
        while (elapsedTime < doorCloseDuration)
        {
            float progress = elapsedTime / doorCloseDuration;
            
            // Move left door to the right (towards center)
            Vector3 leftDoorNewPos = leftDoorCurrentPos + Vector3.right * doorCloseDistance * progress;
            ElevatorL.transform.position = leftDoorNewPos;
            
            if (elapsedTime >= 0.5f && elapsedTime < 0.6f) 
            {
                SwitchToInbetweenCamera();
            }
            // Move right door to the left (towards center)
            Vector3 rightDoorNewPos = rightDoorCurrentPos + Vector3.left * doorCloseDistance * progress;
            ElevatorR.transform.position = rightDoorNewPos;
            
            elapsedTime += Time.deltaTime;
            yield return null;
        }
    }

    private IEnumerator OpenTrapDoor()
    {
        yield return new WaitForSeconds(1f);
        
        Vector3 leftDoorOriginalPos = TrapDoorL.transform.position;
        Vector3 rightDoorOriginalPos = TrapDoorR.transform.position;
        
        float doorOpenDistance = 20f; // How far the doors move apart
        float doorOpenDuration = 2f; // How long the door opening takes
        
        float elapsedTime = 0f;
        
        while (elapsedTime < doorOpenDuration)
        {
            float progress = elapsedTime / doorOpenDuration;
            
            // Move left door to the left
            Vector3 leftDoorNewPos = leftDoorOriginalPos + Vector3.left * doorOpenDistance * progress;
            TrapDoorL.transform.position = leftDoorNewPos;
            
            // Move right door to the right
            Vector3 rightDoorNewPos = rightDoorOriginalPos + Vector3.right * doorOpenDistance * progress;
            TrapDoorR.transform.position = rightDoorNewPos;
            
            elapsedTime += Time.deltaTime;
            yield return null;
        }
        
        // Ensure doors are fully open
        TrapDoorL.transform.position = leftDoorOriginalPos + Vector3.left * doorOpenDistance;
        TrapDoorR.transform.position = rightDoorOriginalPos + Vector3.right * doorOpenDistance;
    }
}
