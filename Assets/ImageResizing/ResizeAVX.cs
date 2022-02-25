using System;
using System.Diagnostics;
using UnityEngine;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Burst.Intrinsics;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;

using static Unity.Burst.Intrinsics.X86.Sse;
using static Unity.Burst.Intrinsics.X86.Sse2;
using static Unity.Burst.Intrinsics.X86.Sse4_1;
using static Unity.Burst.Intrinsics.X86.Avx;
using static Unity.Burst.Intrinsics.X86.Avx2;

public static class ResizeAVX
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
            v256 invOutputWidth = mm256_set1_ps(1.0f / outputWidth);
            v256 invOutputHeight = mm256_set1_ps(1.0f / outputHeight);

            v256 inputWidth = mm256_set1_epi32(this.inputWidth);
            v256 inputHeight = mm256_set1_epi32(this.inputHeight);

            v256 inputWidthF = mm256_cvtepi32_ps(mm256_set1_epi32(this.inputWidth));
            v256 inputHeightF = mm256_cvtepi32_ps(mm256_set1_epi32(this.inputHeight));

            v256 outputWidthF = mm256_cvtepi32_ps(mm256_set1_epi32(this.outputWidth));
            v256 outputHeightF = mm256_cvtepi32_ps(mm256_set1_epi32(this.outputHeight));

            v256 inMaxX = mm256_set1_epi32(this.inputWidth - 1);
            v256 inMaxY = mm256_set1_epi32(this.inputHeight - 1);

            v256 byteMask = mm256_set1_epi32(0xFF);

            Color32* inputPixelsPtr = (Color32*)inputPixels.GetUnsafeReadOnlyPtr();

            v256 x = mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7);
            v256 y = mm256_cvtepi32_ps(mm256_set1_epi32(index));
            int* rowPtr = (int*)outputPixels.GetUnsafePtr() + index * outputWidth;

            while (true)
            {
                v256 u = mm256_mul_ps(mm256_add_ps(x, mm256_set1_ps(0.5f)), invOutputWidth);
                v256 v = mm256_mul_ps(mm256_add_ps(y, mm256_set1_ps(0.5f)), invOutputHeight);

                // sample location in pixel space
                v256 sx = mm256_sub_ps(mm256_mul_ps(u, inputWidthF), mm256_set1_ps(0.5f));
                v256 sy = mm256_sub_ps(mm256_mul_ps(v, inputHeightF), mm256_set1_ps(0.5f));
                v256 floorX = mm256_floor_ps(sx);
                v256 floorY = mm256_floor_ps(sy);

                // interpolator factor
                v256 qx = mm256_sub_ps(sx, floorX);
                v256 qy = mm256_sub_ps(sy, floorY);

                // pixel indices
                v256 lowX = mm256_cvtps_epi32(floorX);
                v256 lowY = mm256_cvtps_epi32(floorY);
                v256 highX = mm256_add_epi32(lowX, mm256_set1_epi32(1));
                v256 highY = mm256_add_epi32(lowY, mm256_set1_epi32(1));

                // TextureWrapMode.Clamp
                lowX = mm256_min_epi32(mm256_max_epi32(lowX, mm256_set1_epi32(0)), inMaxX);
                highX = mm256_min_epi32(mm256_max_epi32(highX, mm256_set1_epi32(0)), inMaxX);
                lowY = mm256_min_epi32(mm256_max_epi32(lowY, mm256_set1_epi32(0)), inMaxY);
                highY = mm256_min_epi32(mm256_max_epi32(highY, mm256_set1_epi32(0)), inMaxY);

                v256 offset11 = mm256_add_epi32(lowX, mm256_mullo_epi32(lowY, inputWidth));
                v256 offset21 = mm256_add_epi32(highX, mm256_mullo_epi32(lowY, inputWidth));
                v256 offset12 = mm256_add_epi32(lowX, mm256_mullo_epi32(highY, inputWidth));
                v256 offset22 = mm256_add_epi32(highX, mm256_mullo_epi32(highY, inputWidth));

#if false
                // This is a lot slower than the dumb code below...
                // Burst is smort :(
                v256 s11 = mm256_i32gather_ps(inputPixelsPtr, offset11, 4);
                v256 s21 = mm256_i32gather_ps(inputPixelsPtr, offset21, 4);
                v256 s12 = mm256_i32gather_ps(inputPixelsPtr, offset12, 4);
                v256 s22 = mm256_i32gather_ps(inputPixelsPtr, offset22, 4);

                v256 s11r = mm256_and_si256(s11, byteMask);
                v256 s11g = mm256_and_si256(mm256_srli_epi32(s11, 8), byteMask);
                v256 s11b = mm256_and_si256(mm256_srli_epi32(s11, 16), byteMask);
                v256 s11a = mm256_srli_epi32(s11, 24);

                v256 s21r = mm256_and_si256(s21, byteMask);
                v256 s21g = mm256_and_si256(mm256_srli_epi32(s21, 8), byteMask);
                v256 s21b = mm256_and_si256(mm256_srli_epi32(s21, 16), byteMask);
                v256 s21a = mm256_srli_epi32(s21, 24);

                v256 s12r = mm256_and_si256(s12, byteMask);
                v256 s12g = mm256_and_si256(mm256_srli_epi32(s12, 8), byteMask);
                v256 s12b = mm256_and_si256(mm256_srli_epi32(s12, 16), byteMask);
                v256 s12a = mm256_srli_epi32(s12, 24);

                v256 s22r = mm256_and_si256(s22, byteMask);
                v256 s22g = mm256_and_si256(mm256_srli_epi32(s22, 8), byteMask);
                v256 s22b = mm256_and_si256(mm256_srli_epi32(s22, 16), byteMask);
                v256 s22a = mm256_srli_epi32(s22, 24);
#else
                // sampling
                Color32 s11_0 = inputPixelsPtr[offset11.UInt0];
                Color32 s21_0 = inputPixelsPtr[offset21.UInt0];
                Color32 s12_0 = inputPixelsPtr[offset12.UInt0];
                Color32 s22_0 = inputPixelsPtr[offset22.UInt0];

                Color32 s11_1 = inputPixelsPtr[offset11.UInt1];
                Color32 s21_1 = inputPixelsPtr[offset21.UInt1];
                Color32 s12_1 = inputPixelsPtr[offset12.UInt1];
                Color32 s22_1 = inputPixelsPtr[offset22.UInt1];

                Color32 s11_2 = inputPixelsPtr[offset11.UInt2];
                Color32 s21_2 = inputPixelsPtr[offset21.UInt2];
                Color32 s12_2 = inputPixelsPtr[offset12.UInt2];
                Color32 s22_2 = inputPixelsPtr[offset22.UInt2];

                Color32 s11_3 = inputPixelsPtr[offset11.UInt3];
                Color32 s21_3 = inputPixelsPtr[offset21.UInt3];
                Color32 s12_3 = inputPixelsPtr[offset12.UInt3];
                Color32 s22_3 = inputPixelsPtr[offset22.UInt3];

                Color32 s11_4 = inputPixelsPtr[offset11.UInt4];
                Color32 s21_4 = inputPixelsPtr[offset21.UInt4];
                Color32 s12_4 = inputPixelsPtr[offset12.UInt4];
                Color32 s22_4 = inputPixelsPtr[offset22.UInt4];

                Color32 s11_5 = inputPixelsPtr[offset11.UInt5];
                Color32 s21_5 = inputPixelsPtr[offset21.UInt5];
                Color32 s12_5 = inputPixelsPtr[offset12.UInt5];
                Color32 s22_5 = inputPixelsPtr[offset22.UInt5];

                Color32 s11_6 = inputPixelsPtr[offset11.UInt6];
                Color32 s21_6 = inputPixelsPtr[offset21.UInt6];
                Color32 s12_6 = inputPixelsPtr[offset12.UInt6];
                Color32 s22_6 = inputPixelsPtr[offset22.UInt6];

                Color32 s11_7 = inputPixelsPtr[offset11.UInt7];
                Color32 s21_7 = inputPixelsPtr[offset21.UInt7];
                Color32 s12_7 = inputPixelsPtr[offset12.UInt7];
                Color32 s22_7 = inputPixelsPtr[offset22.UInt7];

                // interleave samples
                v256 s11r = mm256_setr_epi32(s11_0.r, s11_1.r, s11_2.r, s11_3.r, s11_4.r, s11_5.r, s11_6.r, s11_7.r);
                v256 s11g = mm256_setr_epi32(s11_0.g, s11_1.g, s11_2.g, s11_3.g, s11_4.g, s11_5.g, s11_6.g, s11_7.g);
                v256 s11b = mm256_setr_epi32(s11_0.b, s11_1.b, s11_2.b, s11_3.b, s11_4.b, s11_5.b, s11_6.b, s11_7.b);
                v256 s11a = mm256_setr_epi32(s11_0.a, s11_1.a, s11_2.a, s11_3.a, s11_4.a, s11_5.a, s11_6.a, s11_7.a);

                v256 s21r = mm256_setr_epi32(s21_0.r, s21_1.r, s21_2.r, s21_3.r, s21_4.r, s21_5.r, s21_6.r, s21_7.r);
                v256 s21g = mm256_setr_epi32(s21_0.g, s21_1.g, s21_2.g, s21_3.g, s21_4.g, s21_5.g, s21_6.g, s21_7.g);
                v256 s21b = mm256_setr_epi32(s21_0.b, s21_1.b, s21_2.b, s21_3.b, s21_4.b, s21_5.b, s21_6.b, s21_7.b);
                v256 s21a = mm256_setr_epi32(s21_0.a, s21_1.a, s21_2.a, s21_3.a, s21_4.a, s21_5.a, s21_6.a, s21_7.a);

                v256 s12r = mm256_setr_epi32(s12_0.r, s12_1.r, s12_2.r, s12_3.r, s12_4.r, s12_5.r, s12_6.r, s12_7.r);
                v256 s12g = mm256_setr_epi32(s12_0.g, s12_1.g, s12_2.g, s12_3.g, s12_4.g, s12_5.g, s12_6.g, s12_7.g);
                v256 s12b = mm256_setr_epi32(s12_0.b, s12_1.b, s12_2.b, s12_3.b, s12_4.b, s12_5.b, s12_6.b, s12_7.b);
                v256 s12a = mm256_setr_epi32(s12_0.a, s12_1.a, s12_2.a, s12_3.a, s12_4.a, s12_5.a, s12_6.a, s12_7.a);

                v256 s22r = mm256_setr_epi32(s22_0.r, s22_1.r, s22_2.r, s22_3.r, s22_4.r, s22_5.r, s22_6.r, s22_7.r);
                v256 s22g = mm256_setr_epi32(s22_0.g, s22_1.g, s22_2.g, s22_3.g, s22_4.g, s22_5.g, s22_6.g, s22_7.g);
                v256 s22b = mm256_setr_epi32(s22_0.b, s22_1.b, s22_2.b, s22_3.b, s22_4.b, s22_5.b, s22_6.b, s22_7.b);
                v256 s22a = mm256_setr_epi32(s22_0.a, s22_1.a, s22_2.a, s22_3.a, s22_4.a, s22_5.a, s22_6.a, s22_7.a);
#endif

                // unorm to float conversion
                v256 px11r = UNorm8SrgbToFloat(s11r);
                v256 px11g = UNorm8SrgbToFloat(s11g);
                v256 px11b = UNorm8SrgbToFloat(s11b);
                v256 px11a = UNorm8ToFloat(s11a);

                v256 px21r = UNorm8SrgbToFloat(s21r);
                v256 px21g = UNorm8SrgbToFloat(s21g);
                v256 px21b = UNorm8SrgbToFloat(s21b);
                v256 px21a = UNorm8ToFloat(s21a);

                v256 px12r = UNorm8SrgbToFloat(s12r);
                v256 px12g = UNorm8SrgbToFloat(s12g);
                v256 px12b = UNorm8SrgbToFloat(s12b);
                v256 px12a = UNorm8ToFloat(s12a);

                v256 px22r = UNorm8SrgbToFloat(s22r);
                v256 px22g = UNorm8SrgbToFloat(s22g);
                v256 px22b = UNorm8SrgbToFloat(s22b);
                v256 px22a = UNorm8ToFloat(s22a);

                // lerp in x
                v256 l1r = mm256_add_ps(px11r, mm256_mul_ps(mm256_sub_ps(px21r, px11r), qx));
                v256 l1g = mm256_add_ps(px11g, mm256_mul_ps(mm256_sub_ps(px21g, px11g), qx));
                v256 l1b = mm256_add_ps(px11b, mm256_mul_ps(mm256_sub_ps(px21b, px11b), qx));
                v256 l1a = mm256_add_ps(px11a, mm256_mul_ps(mm256_sub_ps(px21a, px11a), qx));

                v256 l2r = mm256_add_ps(px12r, mm256_mul_ps(mm256_sub_ps(px22r, px12r), qx));
                v256 l2g = mm256_add_ps(px12g, mm256_mul_ps(mm256_sub_ps(px22g, px12g), qx));
                v256 l2b = mm256_add_ps(px12b, mm256_mul_ps(mm256_sub_ps(px22b, px12b), qx));
                v256 l2a = mm256_add_ps(px12a, mm256_mul_ps(mm256_sub_ps(px22a, px12a), qx));

                // lerp in y
                v256 pixelr = mm256_add_ps(l1r, mm256_mul_ps(mm256_sub_ps(l2r, l1r), qy));
                v256 pixelg = mm256_add_ps(l1g, mm256_mul_ps(mm256_sub_ps(l2g, l1g), qy));
                v256 pixelb = mm256_add_ps(l1b, mm256_mul_ps(mm256_sub_ps(l2b, l1b), qy));
                v256 pixela = mm256_add_ps(l1a, mm256_mul_ps(mm256_sub_ps(l2a, l1a), qy));

                // float to unorm conversion
                v256 resr = FloatToUNorm8Srgb(pixelr);
                v256 resg = FloatToUNorm8Srgb(pixelg);
                v256 resb = FloatToUNorm8Srgb(pixelb);
                v256 resa = FloatToUNorm8(pixela);

                v256 store = mm256_and_si256(resr, byteMask);
                store = mm256_or_si256(store, mm256_slli_epi32(resg, 8));
                store = mm256_or_si256(store, mm256_slli_epi32(resb, 16));
                store = mm256_or_si256(store, mm256_slli_epi32(resa, 24));

                v256 mask = mm256_cmpgt_epi32(outputWidthF, x);
                mm256_maskstore_epi32(rowPtr, mask, store);

                if (mm256_movemask_epi8(mask) != -1)
                    break;

                x = mm256_add_ps(x, mm256_set1_ps(8));
                rowPtr += 8;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static v256 UNorm8ToFloat(v256 value)
        {
            const float factor = 1.0f / byte.MaxValue;
            return mm256_mul_ps(mm256_cvtepi32_ps(value), mm256_set1_ps(factor));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static v256 FloatToUNorm8(v256 value)
        {
            // inverting the test maps nan values to 0
            value = mm256_max_ps(value, mm256_set1_ps(0.0f));
            value = mm256_min_ps(value, mm256_set1_ps(1.0f));
            return mm256_cvttps_epi32(mm256_add_ps(mm256_mul_ps(value, mm256_set1_ps(byte.MaxValue)), mm256_set1_ps(0.5f)));
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

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static v256 FloatToUNorm8Srgb(v256 value)
        {
            v256 clampmin = mm256_set1_epi32((127 - 13) << 23);
            v256 almostone = mm256_set1_epi32(0x3f7fffff);
            v256 lutthresh = mm256_set1_epi32(0x3b800000);
            v256 mantmask = mm256_set1_epi32(0xff);
            v256 topscale = mm256_set1_epi32(0x02000000);

            // Clamp to [2^(-13), 1-eps]; these two values map to 0 and 1, respectively.
            // The tests are carefully written so that NaNs map to 0, same as in the reference
            // implementation.
            value = mm256_max_ps(value, clampmin); // written this way to catch NaNs
            value = mm256_min_ps(value, almostone);

            // Do the table lookup and unpack bias, scale
            v256 tabIdx = mm256_srli_epi32(value, 20);

            v256 tabval = mm256_setr_epi32(
                (int)fp32ToSrgb8Tab4[tabIdx.UInt0 - (127-13)*8],
                (int)fp32ToSrgb8Tab4[tabIdx.UInt1 - (127-13)*8],
                (int)fp32ToSrgb8Tab4[tabIdx.UInt2 - (127-13)*8],
                (int)fp32ToSrgb8Tab4[tabIdx.UInt3 - (127-13)*8],
                (int)fp32ToSrgb8Tab4[tabIdx.UInt4 - (127-13)*8],
                (int)fp32ToSrgb8Tab4[tabIdx.UInt5 - (127-13)*8],
                (int)fp32ToSrgb8Tab4[tabIdx.UInt6 - (127-13)*8],
                (int)fp32ToSrgb8Tab4[tabIdx.UInt7 - (127-13)*8]
            );

            v256 tabmult1 = mm256_srli_epi32(value, 12);
            v256 tabmult2 = mm256_and_si256(tabmult1, mantmask);
            v256 tabmult3 = mm256_or_si256(tabmult2, topscale);
            v256 tabprod = mm256_madd_epi16(tabval, tabmult3);
            v256 result = mm256_srli_epi32(tabprod, 16);

            return result;
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
        public static v256 UNorm8SrgbToFloat(v256 value)
        {
            return mm256_setr_ps(
                Srgb8ToF32[value.UInt0],
                Srgb8ToF32[value.UInt1],
                Srgb8ToF32[value.UInt2],
                Srgb8ToF32[value.UInt3],
                Srgb8ToF32[value.UInt4],
                Srgb8ToF32[value.UInt5],
                Srgb8ToF32[value.UInt6],
                Srgb8ToF32[value.UInt7]
            );
        }
    }
}
