using System;
using System.Diagnostics;
using UnityEngine;
using Unity.Burst;
using Unity.Burst.Intrinsics;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;

using static Unity.Burst.Intrinsics.X86.Sse;
using static Unity.Burst.Intrinsics.X86.Sse2;

public static class ResizeVectorized
{
    public static void Resize(NativeArray<Color32> inputPixels, int inputWidth, int inputHeight,
        NativeArray<Color32> outputPixels, int outputWidth, int outputHeight)
    {
        if (inputWidth < 1 || inputHeight < 1)
            throw new ArgumentException("Input width and height must be greater than 0");

        if ((long)inputWidth * (long)inputHeight != inputPixels.Length)
            throw new ArgumentException("Input pixel array must match the size dimensions");

        if (outputWidth < 1 || outputHeight < 1)
            throw new ArgumentException("Output width and height must be greater than 0");

        if ((long)outputWidth * (long)outputHeight != outputPixels.Length)
            throw new ArgumentException("Output pixel array must match the size dimensions");

        ResizeJob job;
        job.inputPixels = inputPixels;
        job.inputWidth = inputWidth;
        job.inputHeight = inputHeight;
        job.outputPixels = outputPixels;
        job.outputWidth = outputWidth;
        job.outputHeight = outputHeight;

        job.Run(outputHeight);
    }

    public static float Benchmark(NativeArray<Color32> inputPixels, int inputWidth, int inputHeight,
        NativeArray<Color32> outputPixels, int outputWidth, int outputHeight, int numIterations)
    {
        if (inputWidth < 1 || inputHeight < 1)
            throw new ArgumentException("Input width and height must be greater than 0");

        if ((long)inputWidth * (long)inputHeight != inputPixels.Length)
            throw new ArgumentException("Input pixel array must match the size dimensions");

        if (outputWidth < 1 || outputHeight < 1)
            throw new ArgumentException("Output width and height must be greater than 0");

        if ((long)outputWidth * (long)outputHeight != outputPixels.Length)
            throw new ArgumentException("Output pixel array must match the size dimensions");

        if (numIterations < 1)
            throw new ArgumentOutOfRangeException(nameof(numIterations), "Number of iterations must be greater than 0");

        var timer = new Stopwatch();

        ResizeJob job;
        job.inputPixels = inputPixels;
        job.inputWidth = inputWidth;
        job.inputHeight = inputHeight;
        job.outputPixels = outputPixels;
        job.outputWidth = outputWidth;
        job.outputHeight = outputHeight;

        // warmup
        for (int i = 0; i < 10; i++)
        {
            job.Run(outputHeight);
        }

        for (int i = 0; i < numIterations; i++)
        {
            timer.Start();
            job.Run(outputHeight);
            timer.Stop();
        }

        return (float)(timer.Elapsed.TotalMilliseconds / numIterations);
    }

    [BurstCompile(CompileSynchronously = true, DisableSafetyChecks = true)]
    unsafe struct ResizeJob : IJobParallelFor
    {
        [ReadOnly, NativeDisableParallelForRestriction]
        public NativeArray<Color32> inputPixels;
        public int inputWidth;
        public int inputHeight;

        [WriteOnly, NativeDisableParallelForRestriction]
        public NativeArray<Color32> outputPixels;
        public int outputWidth;
        public int outputHeight;

        public void Execute(int index)
        {
            int4 iy = new int4(index);
            float4 y = new float4(iy);

            float4 invOutputWidth = new float4(1.0f / outputWidth);
            float4 invOutputHeight = new float4(1.0f / outputHeight);

            Color32* inputPixelsPtr = (Color32*)inputPixels.GetUnsafeReadOnlyPtr();
            Color32* outputPixelsPtr = (Color32*)outputPixels.GetUnsafePtr();

            Color32 store0;
            Color32 store1;
            Color32 store2;
            Color32 store3;
            int4 storeLocation;

            int i = 0;
            while (true)
            {
                int4 ix = new int4(i) + new int4(0, 1, 2, 3);
                float4 x = new float4(ix);

                float4 u = (x + 0.5f) * invOutputWidth;
                float4 v = (y + 0.5f) * invOutputHeight;

                // sample location in pixel space
                float4 sx = u * inputWidth - 0.5f;
                float4 sy = v * inputHeight - 0.5f;
                float4 floorX = math.floor(sx);
                float4 floorY = math.floor(sy);

                // interpolator factor
                float4 qx = sx - floorX;
                float4 qy = sy - floorY;

                // pixel indices
                int4 lowX = (int4)floorX;
                int4 lowY = (int4)floorY;
                int4 highX = lowX + 1;
                int4 highY = lowY + 1;

                // TextureWrapMode.Clamp
                lowX = math.clamp(lowX, 0, inputWidth - 1);
                highX = math.clamp(highX, 0, inputWidth - 1);
                lowY = math.clamp(lowY, 0, inputHeight - 1);
                highY = math.clamp(highY, 0, inputHeight - 1);

                int4 offset11 = lowX + lowY * inputWidth;
                int4 offset21 = highX + lowY * inputWidth;
                int4 offset12 = lowX + highY * inputWidth;
                int4 offset22 = highX + highY * inputWidth;

                // sampling
                Color32 s11_0 = inputPixelsPtr[offset11[0]];
                Color32 s21_0 = inputPixelsPtr[offset21[0]];
                Color32 s12_0 = inputPixelsPtr[offset12[0]];
                Color32 s22_0 = inputPixelsPtr[offset22[0]];

                Color32 s11_1 = inputPixelsPtr[offset11[1]];
                Color32 s21_1 = inputPixelsPtr[offset21[1]];
                Color32 s12_1 = inputPixelsPtr[offset12[1]];
                Color32 s22_1 = inputPixelsPtr[offset22[1]];

                Color32 s11_2 = inputPixelsPtr[offset11[2]];
                Color32 s21_2 = inputPixelsPtr[offset21[2]];
                Color32 s12_2 = inputPixelsPtr[offset12[2]];
                Color32 s22_2 = inputPixelsPtr[offset22[2]];

                Color32 s11_3 = inputPixelsPtr[offset11[3]];
                Color32 s21_3 = inputPixelsPtr[offset21[3]];
                Color32 s12_3 = inputPixelsPtr[offset12[3]];
                Color32 s22_3 = inputPixelsPtr[offset22[3]];

                // interleave samples
                int4 s11r = new int4(s11_0.r, s11_1.r, s11_2.r, s11_3.r);
                int4 s11g = new int4(s11_0.g, s11_1.g, s11_2.g, s11_3.g);
                int4 s11b = new int4(s11_0.b, s11_1.b, s11_2.b, s11_3.b);
                int4 s11a = new int4(s11_0.a, s11_1.a, s11_2.a, s11_3.a);

                int4 s21r = new int4(s21_0.r, s21_1.r, s21_2.r, s21_3.r);
                int4 s21g = new int4(s21_0.g, s21_1.g, s21_2.g, s21_3.g);
                int4 s21b = new int4(s21_0.b, s21_1.b, s21_2.b, s21_3.b);
                int4 s21a = new int4(s21_0.a, s21_1.a, s21_2.a, s21_3.a);

                int4 s12r = new int4(s12_0.r, s12_1.r, s12_2.r, s12_3.r);
                int4 s12g = new int4(s12_0.g, s12_1.g, s12_2.g, s12_3.g);
                int4 s12b = new int4(s12_0.b, s12_1.b, s12_2.b, s12_3.b);
                int4 s12a = new int4(s12_0.a, s12_1.a, s12_2.a, s12_3.a);

                int4 s22r = new int4(s22_0.r, s22_1.r, s22_2.r, s22_3.r);
                int4 s22g = new int4(s22_0.g, s22_1.g, s22_2.g, s22_3.g);
                int4 s22b = new int4(s22_0.b, s22_1.b, s22_2.b, s22_3.b);
                int4 s22a = new int4(s22_0.a, s22_1.a, s22_2.a, s22_3.a);

                // unorm to float conversion
                float4 px11r = UNorm8SrgbToFloat(s11r);
                float4 px11g = UNorm8SrgbToFloat(s11g);
                float4 px11b = UNorm8SrgbToFloat(s11b);
                float4 px11a = UNorm8ToFloat(s11a);

                float4 px21r = UNorm8SrgbToFloat(s21r);
                float4 px21g = UNorm8SrgbToFloat(s21g);
                float4 px21b = UNorm8SrgbToFloat(s21b);
                float4 px21a = UNorm8ToFloat(s21a);

                float4 px12r = UNorm8SrgbToFloat(s12r);
                float4 px12g = UNorm8SrgbToFloat(s12g);
                float4 px12b = UNorm8SrgbToFloat(s12b);
                float4 px12a = UNorm8ToFloat(s12a);

                float4 px22r = UNorm8SrgbToFloat(s22r);
                float4 px22g = UNorm8SrgbToFloat(s22g);
                float4 px22b = UNorm8SrgbToFloat(s22b);
                float4 px22a = UNorm8ToFloat(s22a);

                // lerp in x
                float4 l1r = px11r + (px21r - px11r) * qx;
                float4 l1g = px11g + (px21g - px11g) * qx;
                float4 l1b = px11b + (px21b - px11b) * qx;
                float4 l1a = px11a + (px21a - px11a) * qx;

                float4 l2r = px12r + (px22r - px12r) * qx;
                float4 l2g = px12g + (px22g - px12g) * qx;
                float4 l2b = px12b + (px22b - px12b) * qx;
                float4 l2a = px12a + (px22a - px12a) * qx;

                // lerp in y
                float4 pixelr = l1r + (l2r - l1r) * qy;
                float4 pixelg = l1g + (l2g - l1g) * qy;
                float4 pixelb = l1b + (l2b - l1b) * qy;
                float4 pixela = l1a + (l2a - l1a) * qy;

                // float to unorm conversion
                int4 resr = FloatToUNorm8Srgb(pixelr);
                int4 resg = FloatToUNorm8Srgb(pixelg);
                int4 resb = FloatToUNorm8Srgb(pixelb);
                int4 resa = FloatToUNorm8(pixela);

                store0 = new Color32((byte)resr.x, (byte)resg.x, (byte)resb.x, (byte)resa.x);
                store1 = new Color32((byte)resr.y, (byte)resg.y, (byte)resb.y, (byte)resa.y);
                store2 = new Color32((byte)resr.z, (byte)resg.z, (byte)resb.z, (byte)resa.z);
                store3 = new Color32((byte)resr.w, (byte)resg.w, (byte)resb.w, (byte)resa.w);

                storeLocation = ix + iy * outputWidth;

                // the last store happens outside the loop
                if (i >= outputWidth - 4)
                    break;

                outputPixelsPtr[storeLocation[0]] = store0;
                outputPixelsPtr[storeLocation[1]] = store1;
                outputPixelsPtr[storeLocation[2]] = store2;
                outputPixelsPtr[storeLocation[3]] = store3;

                i += 4;
            }

            // The last store might not be aligned with the simd lane size.
            // Store only pixels that are still in bounds of the output pixels array.
            if (i++ < outputWidth)
                outputPixelsPtr[storeLocation[0]] = store0;
            if (i++ < outputWidth)
                outputPixelsPtr[storeLocation[1]] = store1;
            if (i++ < outputWidth)
                outputPixelsPtr[storeLocation[2]] = store2;
            if (i++ < outputWidth)
                outputPixelsPtr[storeLocation[3]] = store3;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float4 UNorm8ToFloat(int4 value)
        {
            const float factor = 1.0f / byte.MaxValue;
            return value * new float4(factor);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int4 FloatToUNorm8(float4 value)
        {
            // inverting the test maps nan values to 0
            value = math.max(value, 0.0f);
            value = math.min(value, 1.0f);
            return (int4)(value * byte.MaxValue + 0.5f);
        }

        // sRGB conversion from https://gist.github.com/rygorous/2203834

        static readonly uint[] fp32ToSrgb8Tab4 = new uint[104]
        {
            0x0073000d, 0x007a000d, 0x0080000d, 0x0087000d, 0x008d000d, 0x0094000d, 0x009a000d, 0x00a1000d,
            0x00a7001a, 0x00b4001a, 0x00c1001a, 0x00ce001a, 0x00da001a, 0x00e7001a, 0x00f4001a, 0x0101001a,
            0x010e0033, 0x01280033, 0x01410033, 0x015b0033, 0x01750033, 0x018f0033, 0x01a80033, 0x01c20033,
            0x01dc0067, 0x020f0067, 0x02430067, 0x02760067, 0x02aa0067, 0x02dd0067, 0x03110067, 0x03440067,
            0x037800ce, 0x03df00ce, 0x044600ce, 0x04ad00ce, 0x051400ce, 0x057b00c5, 0x05dd00bc, 0x063b00b5,
            0x06970158, 0x07420142, 0x07e30130, 0x087b0120, 0x090b0112, 0x09940106, 0x0a1700fc, 0x0a9500f2,
            0x0b0f01cb, 0x0bf401ae, 0x0ccb0195, 0x0d950180, 0x0e56016e, 0x0f0d015e, 0x0fbc0150, 0x10630143,
            0x11070264, 0x1238023e, 0x1357021d, 0x14660201, 0x156601e9, 0x165a01d3, 0x174401c0, 0x182401af,
            0x18fe0331, 0x1a9602fe, 0x1c1502d2, 0x1d7e02ad, 0x1ed4028d, 0x201a0270, 0x21520256, 0x227d0240,
            0x239f0443, 0x25c003fe, 0x27bf03c4, 0x29a10392, 0x2b6a0367, 0x2d1d0341, 0x2ebe031f, 0x304d0300,
            0x31d105b0, 0x34a80555, 0x37520507, 0x39d504c5, 0x3c37048b, 0x3e7c0458, 0x40a8042a, 0x42bd0401,
            0x44c20798, 0x488e071e, 0x4c1c06b6, 0x4f76065d, 0x52a50610, 0x55ac05cc, 0x5892058f, 0x5b590559,
            0x5e0c0a23, 0x631c0980, 0x67db08f6, 0x6c55087f, 0x70940818, 0x74a007bd, 0x787d076c, 0x7c330723,
        };

        [StructLayout(LayoutKind.Explicit)]
        unsafe struct FP32
        {
            [FieldOffset(0)]
            public uint u;
            [FieldOffset(0)]
            public float f;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int4 FloatToUNorm8Srgb(float4 value)
        {
            if (X86.Sse2.IsSse2Supported)
            {
                v128 val = new v128(value.x, value.y, value.z, value.w);

                v128 clampmin = set1_epi32((127 - 13) << 23);
                v128 almostone = set1_epi32(0x3f7fffff);
                v128 lutthresh = set1_epi32(0x3b800000);
                v128 mantmask = set1_epi32(0xff);
                v128 topscale = set1_epi32(0x02000000);

                // Clamp to [2^(-13), 1-eps]; these two values map to 0 and 1, respectively.
                // The tests are carefully written so that NaNs map to 0, same as in the reference
                // implementation.
                val = max_ps(val, clampmin); // written this way to catch NaNs
                val = min_ps(val, almostone);

                // Do the table lookup and unpack bias, scale
                v128 tabIdx = srli_epi32(val, 20);

                v128 tabval = setr_epi32(
                    (int)fp32ToSrgb8Tab4[tabIdx.UInt0 - (127 - 13) * 8],
                    (int)fp32ToSrgb8Tab4[tabIdx.UInt1 - (127 - 13) * 8],
                    (int)fp32ToSrgb8Tab4[tabIdx.UInt2 - (127 - 13) * 8],
                    (int)fp32ToSrgb8Tab4[tabIdx.UInt3 - (127 - 13) * 8]
                );

                v128 tabmult1 = srli_epi32(val, 12);
                v128 tabmult2 = and_si128(tabmult1, mantmask);
                v128 tabmult3 = or_si128(tabmult2, topscale);
                v128 tabprod = madd_epi16(tabval, tabmult3);
                v128 result = srli_epi32(tabprod, 16);

                return new int4((int)result.UInt0, (int)result.UInt1, (int)result.UInt2, (int)result.UInt3);
            }
            else
            {
                float4 almostone = math.asfloat(new uint4(0x3f7fffff));
                float4 minval = math.asfloat(new uint4((127 - 13) << 23));

                // Clamp to [2^(-13), 1-eps]; these two values map to 0 and 1, respectively.
                // The tests are carefully written so that NaNs map to 0, same as in the reference
                // implementation.
                value = math.max(value, minval); // written this way to catch NaNs
                value = math.min(value, almostone);

                // Do the table lookup and unpack bias, scale
                uint4 valueU = math.asuint(value);
                uint4 tabIdx = (valueU - math.asuint(minval)) >> 20;
                uint4 tab = new uint4(
                    fp32ToSrgb8Tab4[tabIdx[0]],
                    fp32ToSrgb8Tab4[tabIdx[1]],
                    fp32ToSrgb8Tab4[tabIdx[2]],
                    fp32ToSrgb8Tab4[tabIdx[3]]
                );
                uint4 bias = (tab >> 16) << 9;
                uint4 scale = tab & 0xffff;

                // Grab next-highest mantissa bits and perform linear interpolation
                uint4 t = (valueU >> 12) & 0xff;
                return (int4)((bias + scale * t) >> 16);
            }
        }

        static readonly float[] Srgb8ToF32 = new float[256]
        {
            0.000000f, 0.000304f, 0.000607f, 0.000911f, 0.001214f, 0.001518f, 0.001821f, 0.002125f, 0.002428f, 0.002732f, 0.003035f,
            0.003347f, 0.003677f, 0.004025f, 0.004391f, 0.004777f, 0.005182f, 0.005605f, 0.006049f, 0.006512f, 0.006995f, 0.007499f,
            0.008023f, 0.008568f, 0.009134f, 0.009721f, 0.010330f, 0.010960f, 0.011612f, 0.012286f, 0.012983f, 0.013702f, 0.014444f,
            0.015209f, 0.015996f, 0.016807f, 0.017642f, 0.018500f, 0.019382f, 0.020289f, 0.021219f, 0.022174f, 0.023153f, 0.024158f,
            0.025187f, 0.026241f, 0.027321f, 0.028426f, 0.029557f, 0.030713f, 0.031896f, 0.033105f, 0.034340f, 0.035601f, 0.036889f,
            0.038204f, 0.039546f, 0.040915f, 0.042311f, 0.043735f, 0.045186f, 0.046665f, 0.048172f, 0.049707f, 0.051269f, 0.052861f,
            0.054480f, 0.056128f, 0.057805f, 0.059511f, 0.061246f, 0.063010f, 0.064803f, 0.066626f, 0.068478f, 0.070360f, 0.072272f,
            0.074214f, 0.076185f, 0.078187f, 0.080220f, 0.082283f, 0.084376f, 0.086500f, 0.088656f, 0.090842f, 0.093059f, 0.095307f,
            0.097587f, 0.099899f, 0.102242f, 0.104616f, 0.107023f, 0.109462f, 0.111932f, 0.114435f, 0.116971f, 0.119538f, 0.122139f,
            0.124772f, 0.127438f, 0.130136f, 0.132868f, 0.135633f, 0.138432f, 0.141263f, 0.144128f, 0.147027f, 0.149960f, 0.152926f,
            0.155926f, 0.158961f, 0.162029f, 0.165132f, 0.168269f, 0.171441f, 0.174647f, 0.177888f, 0.181164f, 0.184475f, 0.187821f,
            0.191202f, 0.194618f, 0.198069f, 0.201556f, 0.205079f, 0.208637f, 0.212231f, 0.215861f, 0.219526f, 0.223228f, 0.226966f,
            0.230740f, 0.234551f, 0.238398f, 0.242281f, 0.246201f, 0.250158f, 0.254152f, 0.258183f, 0.262251f, 0.266356f, 0.270498f,
            0.274677f, 0.278894f, 0.283149f, 0.287441f, 0.291771f, 0.296138f, 0.300544f, 0.304987f, 0.309469f, 0.313989f, 0.318547f,
            0.323143f, 0.327778f, 0.332452f, 0.337164f, 0.341914f, 0.346704f, 0.351533f, 0.356400f, 0.361307f, 0.366253f, 0.371238f,
            0.376262f, 0.381326f, 0.386430f, 0.391573f, 0.396755f, 0.401978f, 0.407240f, 0.412543f, 0.417885f, 0.423268f, 0.428691f,
            0.434154f, 0.439657f, 0.445201f, 0.450786f, 0.456411f, 0.462077f, 0.467784f, 0.473532f, 0.479320f, 0.485150f, 0.491021f,
            0.496933f, 0.502887f, 0.508881f, 0.514918f, 0.520996f, 0.527115f, 0.533276f, 0.539480f, 0.545725f, 0.552011f, 0.558340f,
            0.564712f, 0.571125f, 0.577581f, 0.584078f, 0.590619f, 0.597202f, 0.603827f, 0.610496f, 0.617207f, 0.623960f, 0.630757f,
            0.637597f, 0.644480f, 0.651406f, 0.658375f, 0.665387f, 0.672443f, 0.679543f, 0.686685f, 0.693872f, 0.701102f, 0.708376f,
            0.715694f, 0.723055f, 0.730461f, 0.737911f, 0.745404f, 0.752942f, 0.760525f, 0.768151f, 0.775822f, 0.783538f, 0.791298f,
            0.799103f, 0.806952f, 0.814847f, 0.822786f, 0.830770f, 0.838799f, 0.846873f, 0.854993f, 0.863157f, 0.871367f, 0.879622f,
            0.887923f, 0.896269f, 0.904661f, 0.913099f, 0.921582f, 0.930111f, 0.938686f, 0.947307f, 0.955974f, 0.964686f, 0.973445f,
            0.982251f, 0.991102f, 1.0f
        };

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float4 UNorm8SrgbToFloat(int4 value)
        {
            return new float4(
                Srgb8ToF32[value[0]],
                Srgb8ToF32[value[1]],
                Srgb8ToF32[value[2]],
                Srgb8ToF32[value[3]]
            );
        }
    }
}
