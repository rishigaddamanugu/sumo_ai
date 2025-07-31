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
        yield return new WaitForSeconds(2f);
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
        Rigidbody elevatorRb = Elevator.GetComponent<Rigidbody>();
        if (elevatorRb == null)
        {
            Debug.LogWarning("Elevator needs a Rigidbody for rumble to work with physics!");
            yield break;
        }
        
        float rumbleDuration = 8f;
        float rumbleIntensity = 0.2f;
        float rumbleSpeed = 20f;
        
        float elapsedTime = 0f;
        
        while (elapsedTime < rumbleDuration)
        {
            // Apply rumble as force instead of setting position
            float xForce = Mathf.Sin(Time.time * rumbleSpeed) * rumbleIntensity;
            float yForce = Mathf.Cos(Time.time * rumbleSpeed * 0.7f) * rumbleIntensity * 0.5f;
            float zForce = Mathf.Sin(Time.time * rumbleSpeed * 1.3f) * rumbleIntensity * 0.3f;
            
            elevatorRb.AddForce(new Vector3(xForce, yForce, zForce), ForceMode.Impulse);
            
            elapsedTime += Time.deltaTime;
            yield return null;
        }
    }
}
