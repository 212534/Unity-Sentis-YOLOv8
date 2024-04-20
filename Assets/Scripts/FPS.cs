using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class FPS : MonoBehaviour
{
    public Text fpsText;
    private float fps;
    private float intervalTime = 0.12f;

    void Update()
    {
        intervalTime -= Time.deltaTime;
        if(intervalTime <0)
        {
            intervalTime = Time.deltaTime + 0.2f;
            ModefyFps();
        }
        
        fps = 1.0f / Time.deltaTime;
    }

    public void ModefyFps()
    {
        fpsText.text = string.Format("FPS: {0:.0f}", fps);
    }

    
}
