using UnityEngine;


public class navbar : MonoBehaviour
{
    private static navbar instance;

    void Awake()
    {
        if (instance != null)
        {
            Destroy(gameObject); // Keeps this object across scene loads
        }
        else
        {
            instance = this; // Destroy duplicates
        }
        DontDestroyOnLoad(gameObject);
    }
}
