using UnityEngine;
using System.Collections;

public class SceneController : MonoBehaviour
{
    public GameObject Agent;
    public GameObject Elevator;
    public GameObject Platform;
    public GameObject Stadium;
    public GameObject CubeCamera;
    public GameObject ElevatorCamera;
    public GameObject CubeSideCamera;
    public GameObject CubeRevolveCamera;
    public GameObject Scoreboard;
    public GameObject ElevatorL;
    public GameObject ElevatorR;
    
    void Start()
    {
        StartCoroutine(ExecuteSceneBits());
    }
    
    private IEnumerator ExecuteSceneBits()
    {
        yield return StartCoroutine(SceneBit1());
        yield return StartCoroutine(SceneBit2());
        yield return StartCoroutine(SceneBit3());
        yield return StartCoroutine(SceneBit4());
        yield return StartCoroutine(SceneBit5());
    }
    
    private IEnumerator SceneBit1()
    {
        SwitchToCubeCamera();
        // Cube is shown in elevator, and it is very clear by the shaking that elevator is moving
        Debug.Log("Scene Bit 1: Starting");
        
        // Start gentle elevator rumble
        StartCoroutine(RumbleElevator());
        
        yield return new WaitForSeconds(2f);
        Debug.Log("Scene Bit 1: Completed");
    }

    
    private IEnumerator SceneBit2()
    {
        // Elevator doors open (with a ding) and the cube slides out of the elevator
        Debug.Log("Scene Bit 2: Starting");
        
        // Switch to elevator camera
        SwitchToElevatorCamera();
        
        // Wait 1 second
        yield return new WaitForSeconds(1f);
        
        // Open elevator doors
        yield return StartCoroutine(OpenElevatorDoors());
        
        Debug.Log("Scene Bit 2: Completed");
    }

    private IEnumerator SceneBit3()
    {
        // Cube slides over to the platform, and hops on
        Debug.Log("Scene Bit 3: Starting");
        yield return new WaitForSeconds(2f);
        Debug.Log("Scene Bit 3: Completed");
    }
    
    private IEnumerator SceneBit4()
    {
        // Stadium and camera rumble, and scoreboard goes up
        Debug.Log("Scene Bit 4: Starting");
        yield return new WaitForSeconds(2f);
        Debug.Log("Scene Bit 4: Completed");
    }

    private IEnumerator SceneBit5()
    {
        // Stadium rumbles, camera shakes, and platform rises
        Debug.Log("Scene Bit 5: Starting");
        yield return new WaitForSeconds(2f);
        Debug.Log("Scene Bit 5: Completed");
    }

    private IEnumerator RumbleElevator()
    {
        Vector3 originalPosition = Elevator.transform.position;
        float rumbleDuration = 5f;
        float rumbleIntensity = 0.2f; // Gentle rumble intensity
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
    }
    
    private IEnumerator OpenElevatorDoors()
    {
        Vector3 leftDoorOriginalPos = ElevatorL.transform.position;
        Vector3 rightDoorOriginalPos = ElevatorR.transform.position;
        
        float doorOpenDistance = 2f; // How far the doors move apart
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
}
