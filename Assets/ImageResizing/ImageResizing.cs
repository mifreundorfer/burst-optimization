using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using Unity.Collections;

public class ImageResizing : MonoBehaviour
{
    enum Mode
    {
        Scalar,
        ScalarUnsafe,
        Vectorized,
        SSE,
    }

    [SerializeField]
    Texture2D inputTexture;

    [SerializeField]
    int outputWidth = 900;

    [SerializeField]
    int outputHeight = 825;

    [SerializeField]
    int benchmarkIterations = 100;

    Texture2D outputTexture;

    Mode currentMode = Mode.Scalar;

    float benchmarkResult;

    void OnEnable()
    {
        outputTexture = new Texture2D(outputWidth, outputHeight, GraphicsFormat.R8G8B8A8_SRGB, TextureCreationFlags.None)
        {
            filterMode = FilterMode.Bilinear,
            wrapMode = TextureWrapMode.Clamp,
        };

        ResizeImage();
    }

    void ResizeImage()
    {
        using NativeArray<Color32> inputPixels = new NativeArray<Color32>(inputTexture.GetPixels32(), Allocator.TempJob);
        using NativeArray<Color32> outputPixels = new NativeArray<Color32>(outputWidth * outputHeight, Allocator.TempJob);

        switch (currentMode)
        {
            case Mode.Scalar:
                ResizeScalar.Resize(inputPixels, inputTexture.width, inputTexture.height,
                    outputPixels, outputWidth, outputHeight);
                break;

            case Mode.ScalarUnsafe:
                ResizeScalarUnsafe.Resize(inputPixels, inputTexture.width, inputTexture.height,
                    outputPixels, outputWidth, outputHeight);
                break;

            case Mode.Vectorized:
                ResizeVectorized.Resize(inputPixels, inputTexture.width, inputTexture.height,
                    outputPixels, outputWidth, outputHeight);
                break;

            case Mode.SSE:
                ResizeSSE.Resize(inputPixels, inputTexture.width, inputTexture.height,
                    outputPixels, outputWidth, outputHeight);
                break;

            default:
                throw new Exception("Invalid Mode enum");
        }

        outputTexture.SetPixelData(outputPixels, 0);
        outputTexture.Apply(false, false);
    }

    void RunBenchmark()
    {
        using NativeArray<Color32> inputPixels = new NativeArray<Color32>(inputTexture.GetPixels32(), Allocator.TempJob);
        using NativeArray<Color32> outputPixels = new NativeArray<Color32>(outputWidth * outputHeight, Allocator.TempJob);

        switch (currentMode)
        {
            case Mode.Scalar:
                benchmarkResult = ResizeScalar.Benchmark(inputPixels, inputTexture.width, inputTexture.height,
                    outputPixels, outputWidth, outputHeight, benchmarkIterations);
                break;

            case Mode.ScalarUnsafe:
                benchmarkResult = ResizeScalarUnsafe.Benchmark(inputPixels, inputTexture.width, inputTexture.height,
                    outputPixels, outputWidth, outputHeight, benchmarkIterations);
                break;

            case Mode.Vectorized:
                benchmarkResult = ResizeVectorized.Benchmark(inputPixels, inputTexture.width, inputTexture.height,
                    outputPixels, outputWidth, outputHeight, benchmarkIterations);
                break;

            case Mode.SSE:
                benchmarkResult = ResizeSSE.Benchmark(inputPixels, inputTexture.width, inputTexture.height,
                    outputPixels, outputWidth, outputHeight, benchmarkIterations);
                break;

            default:
                throw new Exception("Invalid Mode enum");
        }
    }

    void OnGUI()
    {
        Rect rect = new Rect(0, 0, Screen.width, Screen.height);

        {
            Rect headerRect = rect.RemoveFromTop(20);

            GUILayout.BeginArea(headerRect);
            GUILayout.BeginHorizontal();

            foreach (Mode mode in Enum.GetValues(typeof(Mode)))
            {
                string name = Enum.GetName(typeof(Mode), mode);
                if (GUILayout.Button(name))
                {
                    currentMode = mode;
                    ResizeImage();
                }
            }

            GUILayout.EndHorizontal();
            GUILayout.EndArea();
        }

        rect.RemoveFromTop(8);

        {
            Rect descriptionRect = rect.RemoveFromTop(20);

            GUILayout.BeginArea(descriptionRect);
            GUILayout.BeginHorizontal();

            GUILayout.Label($"Current Mode: {currentMode}");
            if (GUILayout.Button("Benchmark"))
            {
                RunBenchmark();
            }

            GUILayout.Label($"Benchmark Result: {benchmarkResult:f1}ms");

            GUILayout.EndHorizontal();
            GUILayout.EndArea();
        }

        rect.RemoveFromTop(8);

        {
            Rect leftViewRect = rect.RemoveFromLeft(Mathf.Round(rect.width * 0.5f));
            Rect rightViewRect = rect;

            Color backgroundColor = new Color32(20, 20, 20, 255);
            DrawRect(leftViewRect, backgroundColor);
            DrawRect(rightViewRect, backgroundColor);

            leftViewRect = leftViewRect.Adjusted(-10, -30, -10, -10);
            rightViewRect = rightViewRect.Adjusted(-10, -30, -10, -10);

            GUI.Label(new Rect(leftViewRect.x, leftViewRect.y - 20, leftViewRect.width, 20), "Input Image");
            GUI.Label(new Rect(rightViewRect.x, rightViewRect.y - 20, rightViewRect.width, 20), "Output Image");

            if (inputTexture != null)
            {
                float width = inputTexture.width;
                float height = inputTexture.height;

                float ratio = Mathf.Max(width / leftViewRect.width, height / leftViewRect.height);
                if (ratio > 1.0f)
                {
                    float factor = 1.0f / ratio;
                    width = Mathf.Round(width * factor);
                    height = Mathf.Round(height * factor);
                }

                Rect leftImageRect = leftViewRect.CenteredRect(new Vector2(width, height));
                GUI.DrawTexture(leftImageRect, inputTexture, ScaleMode.StretchToFill, false);
            }

            if (outputTexture != null)
            {
                float width = outputTexture.width;
                float height = outputTexture.height;

                float ratio = Mathf.Max(width / rightViewRect.width, height / rightViewRect.height);
                if (ratio > 1.0f)
                {
                    float factor = 1.0f / ratio;
                    width = Mathf.Round(width * factor);
                    height = Mathf.Round(height * factor);
                }

                Rect rightImageRect = rightViewRect.CenteredRect(new Vector2(width, height));
                GUI.DrawTexture(rightImageRect, outputTexture, ScaleMode.StretchToFill, false);
            }
        }
    }

    static void DrawRect(Rect rect, Color color)
    {
        Color oldColor = GUI.color;
        GUI.color = color;
        GUI.DrawTexture(rect, Texture2D.whiteTexture);
        GUI.color = oldColor;
    }
}
