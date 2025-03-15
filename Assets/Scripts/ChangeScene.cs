using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using UnityEngine.SceneManagement;




public class ChangeScene : MonoBehaviour
{
    public void moveToScene(int sceneID){
        SceneManager.LoadScene(sceneID);
    }
}
