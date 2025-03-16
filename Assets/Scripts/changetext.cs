using UnityEngine;
using UnityEngine.UI;
using static User;
using System.Collections.Generic;
public class changetext : MonoBehaviour
{
    public Text text;
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        text.text = User.points.ToString();
    }
}
