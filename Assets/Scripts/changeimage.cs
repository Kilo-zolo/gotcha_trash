
using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using UnityEngine.UI;

public class changeimage: MonoBehaviour
{
    public Image oldImage;
    public Sprite newImage;
    public void ImageChange(){
        oldImage.sprite = newImage;
    }
}
