[System.Serializable]
public class StateRequest
{
    public float[] state;
    public float reward;
}

[System.Serializable]
public class DirectionResponse
{
    public string action;
}
