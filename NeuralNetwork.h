// Fill out your copyright notice in the Description page of Project Settings.
#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Engine/TextureRenderTarget2D.h"

// Import NNE Includes
#include "NNE.h"
#include "NNERuntimeCPU.h"
#include "NNEModelData.h"
#include "Async/Async.h"

#include "NeuralNetwork.generated.h"

DECLARE_DYNAMIC_DELEGATE_OneParam(FNNEAsyncInferenceDelegate, const TArray<float>&, OutData);

// Helper class to store model-related data and operations
class FModelHelper
{
public:
    TUniquePtr<UE::NNE::IModelInstanceCPU> ModelInstance;
    TArray<float> InputData;
    TArray<float> OutputData;
    TArray<UE::NNE::FTensorBindingCPU> InputBindings;
    TArray<UE::NNE::FTensorBindingCPU> OutputBindings;
    bool bIsRunning;
};

UCLASS()
class AI_PLAYGROUND_API ANeuralNetwork : public AActor
{
    GENERATED_BODY()

public:
    // Sets default values for this actor's properties
    ANeuralNetwork();

    // Model Creation
    UFUNCTION(BlueprintCallable, Category = "NNE Neural Network")
    void LoadModelDataAsync();

    UFUNCTION(BlueprintCallable, Category = "NNE Neural Network")
    bool CreateCPUModel();

    // Model Info
    // Getters
    UFUNCTION(BlueprintCallable, Category = "NNE Neural Network")
    void GetInputTensorDescs(bool isModelRunning, int32& numInputs, int32& idxInputs);

    UFUNCTION(BlueprintCallable, Category = "NNE Neural Network")
    void GetInputTensorShape(int32 InputIdx, bool isModelRunning, int32& Rank, int32& Volume, int32& Dimension, int32& Frame, int32& ColorChannels, int32& Height, int32& Width);

    UFUNCTION(BlueprintCallable, Category = "NNE Neural Network")
    int32 GetOutputTensorDescs(bool isModelRunning);

    UFUNCTION(BlueprintCallable, Category = "NNE Neural Network")
    void GetOutputTensorShape(int32 OutputIdx, bool isModelRunning, int32& Rank, int32& Volume, int32& Dimension, int32& PredOpts);

    // Setters
    UFUNCTION(BlueprintCallable, Category = "NNE Neural Network")
    void SetInputTensorShapes(bool isModelRunning, int32 Rank, int32 Dimension, int32 Frame, int32 ColorChannels, int32 Height, int32 Width);

    UFUNCTION(BlueprintCallable, Category = "NNE Neural Network")
    void CreateInputTensorBinding(bool isModelRunning, bool& InBSuccess);

    UFUNCTION(BlueprintCallable, Category = "NNE Neural Network")
    void CreateOutputTensorBinding(bool isModelRunning, bool& OutBSuccess);

    // Image Capture Method
    UFUNCTION(BlueprintCallable, Category = "NNE Data ImageCapture")
    void RT2PixelBuffer(UTextureRenderTarget2D* InputRT, TArray<FColor>& ImagePixelBuffer, int32& OriginalHeight, int32& OriginalWidth);


    // Data PreProcessing
    UFUNCTION(BlueprintCallable, Category = "NNE Data PreProcessing")
    void ResizeImage(TArray<FColor> ImagePixelBuffer, int32 OriginalHeight, int32 OriginalWidth, int32 ResizeHeight, int32 ResizeWidth, TArray<FLinearColor>& ResizedPixelBuffer, int32& ResizedHeight, int32& ResizedWidth);

    UFUNCTION(BlueprintCallable, Category = "NNE Data PreProcessing")
    void NormalizeImage(TArray<FLinearColor> ImageBuffer, int32 ColorChannels, TArray<float>& FlatImg);

    // Properties
    UPROPERTY(EditAnywhere)
    TSoftObjectPtr<UNNEModelData> LazyLoadedModelData;

    UPROPERTY(BlueprintReadOnly, Category = "NNE Neural Network")
    bool IsModelRunning = false;

    // Events
    UFUNCTION(BlueprintImplementableEvent, Category = "NNE Inference")
    void OnModelDataLoaded();

    // Async
    UFUNCTION(BlueprintCallable, Category = "NNE Inference")
    void RunAsyncInference(bool InBSuccess, bool OutBSuccess, TArray<float> InputData, FNNEAsyncInferenceDelegate Result);

private:

    // Controlled Access
    FCriticalSection m_mutex;

    // Model Base
    TSharedPtr<FModelHelper> m_ModelHelper;

    TUniquePtr<UE::NNE::IModelCPU> Model;

    // Model Input Info
    TConstArrayView<UE::NNE::FTensorDesc> InputTensorDescs;

    UE::NNE::FSymbolicTensorShape SymbolicInputTensorShape;

    TArray<UE::NNE::FTensorShape> InputTensorShapes;

    // Model Output Info
    TConstArrayView<UE::NNE::FTensorDesc> OutputTensorDescs;

    UE::NNE::FSymbolicTensorShape SymbolicOutputTensorShape;

    TArray<UE::NNE::FTensorShape> OutputTensorShapes;

    TArray<float> tempModelInput;
};
