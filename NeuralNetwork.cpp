#include "NeuralNetwork.h"
#include "ImageUtils.h"
#include "Engine/AssetManager.h"


ANeuralNetwork::ANeuralNetwork()
{
    // Set this actor to call Tick() every frame. You can turn this off to improve performance if you don't need it.
    PrimaryActorTick.bCanEverTick = true;
}


// Asynchronously loads model data and calls OnModelDataLoaded when complete.
void ANeuralNetwork::LoadModelDataAsync()
{
    if (LazyLoadedModelData.IsNull())
    {
        UE_LOG(LogTemp, Error, TEXT("LazyLoadedModelData is not set, please assign it in the editor"));
    }
    else
    {
        // Async Load Model Data and call OnModelDataLoaded when complete
        UAssetManager::GetStreamableManager().RequestAsyncLoad(LazyLoadedModelData.ToSoftObjectPath(), FStreamableDelegate::CreateUObject(this, &ANeuralNetwork::OnModelDataLoaded));
    }
}


// Create Model and set the Model Helper
bool ANeuralNetwork::CreateCPUModel()
{
    bool bSuccess = false;

      // Check that model data is loaded and valid
    if (LazyLoadedModelData.IsValid())
    {
        // Log the name of the loaded model
        UE_LOG(LogTemp, Display, TEXT("LazyLoadedModelData loaded %s"), *LazyLoadedModelData.Get()->GetName());

        // Create NNE Runtime
        TWeakInterfacePtr<INNERuntimeCPU> Runtime = UE::NNE::GetRuntime<INNERuntimeCPU>(FString("NNERuntimeORTCpu"));
        if (Runtime.IsValid())
        {
            // Initialize ModelHelper and create the model instance
            m_ModelHelper = MakeShared<FModelHelper>();
            Model = Runtime->CreateModel(LazyLoadedModelData.Get());

            if (Model.IsValid())
            {
                m_mutex.Lock();

                m_ModelHelper->ModelInstance = Model->CreateModelInstance();

                m_mutex.Unlock();

                if (m_ModelHelper->ModelInstance.IsValid())
                {
                    // Model creation successful
                    m_mutex.Lock();
                    m_ModelHelper->bIsRunning = false;
                    m_mutex.Unlock();

                    IsModelRunning = false;
                    bSuccess = true;
                }
                else
                {
                    UE_LOG(LogTemp, Display, TEXT("Failed to create Model Instance"));
                }
            }
            else
            {
                UE_LOG(LogTemp, Display, TEXT("Failed to create Model"));
            }
        }
        else
        {
            UE_LOG(LogTemp, Display, TEXT("Failed to create Runtime"));
        }
    }
    return bSuccess;
}


// Retrieves input tensor descriptors from the model instance.
void ANeuralNetwork::GetInputTensorDescs(bool isModelRunning, int32& numInputs, int32& idxInputs)
{
    int32 Inputs;

    if (m_ModelHelper->ModelInstance.IsValid())
    {
        // Stop the model if it's running
        if (isModelRunning)
        {
            m_mutex.Lock();
            m_ModelHelper->bIsRunning = false;
            m_mutex.Unlock();
            IsModelRunning = false;
        }

        // Get input tensor descriptors
        m_mutex.Lock();
        InputTensorDescs = m_ModelHelper->ModelInstance->GetInputTensorDescs();
        m_mutex.Unlock();

        checkf(InputTensorDescs.Num() == 1, TEXT("The current example supports only models with a single input tensor"));

        Inputs = InputTensorDescs.Num();
    }

    numInputs = Inputs;
    idxInputs = Inputs - 1;
}


// Retrieves input tensor shape from the model instance.
void ANeuralNetwork::GetInputTensorShape(int32 InputIdx, bool isModelRunning, int32& Rank, int32& Volume, int32& Dimension, int32& Frame, int32& ColorChannels, int32& Height, int32& Width)
{
    if (m_ModelHelper->ModelInstance.IsValid())
    {
        // Stop the model if it's running
        if (isModelRunning)
        {
            m_mutex.Lock();
            m_ModelHelper->bIsRunning = false;
            m_mutex.Unlock();
            IsModelRunning = false;
        }

        // Get symbolic input tensor shape
        SymbolicInputTensorShape = InputTensorDescs[InputIdx].GetShape();

        checkf(SymbolicInputTensorShape.IsConcrete(), TEXT("The current example supports only models without variable input tensor dimensions"));

        InputTensorShapes = { UE::NNE::FTensorShape::MakeFromSymbolic(SymbolicInputTensorShape) };
    }
    else
    {
        UE_LOG(LogTemp, Display, TEXT("No Valid Model Instance in Model Helper"));
    }

    Rank = InputTensorShapes[InputIdx].Rank();
    Volume = InputTensorShapes[InputIdx].Volume();
    Dimension = InputTensorShapes[InputIdx].GetData()[0];

    if (Rank == 4)
    {
        Frame = 0;
        ColorChannels = InputTensorShapes[InputIdx].GetData()[1];
        Height = InputTensorShapes[InputIdx].GetData()[2];
        Width = InputTensorShapes[InputIdx].GetData()[3];
    }
    if (Rank == 5)
    {
        Frame = InputTensorShapes[InputIdx].GetData()[1];
        ColorChannels = InputTensorShapes[InputIdx].GetData()[2];
        Height = InputTensorShapes[InputIdx].GetData()[3];
        Width = InputTensorShapes[InputIdx].GetData()[4];
    }
}


// Retrieves output tensor descriptors from the model instance.
int32 ANeuralNetwork::GetOutputTensorDescs(bool isModelRunning)
{
    int32 Outputs;

    if (m_ModelHelper->ModelInstance.IsValid())
    {
        // Stop the model if it's running
        if (isModelRunning)
        {
            m_mutex.Lock();
            m_ModelHelper->bIsRunning = false;
            m_mutex.Unlock();
            IsModelRunning = false;
        }

        // Get output tensor descriptors
        m_mutex.Lock();
        OutputTensorDescs = m_ModelHelper->ModelInstance->GetOutputTensorDescs();
        m_mutex.Unlock();

        checkf(OutputTensorDescs.Num() == 1, TEXT("The current example supports only models with a single output tensor"));

        Outputs = OutputTensorDescs.Num();
    }

    return Outputs;
}


// Retrieves output tensor shape from the model instance.
void ANeuralNetwork::GetOutputTensorShape(int32 OutputIdx, bool isModelRunning, int32& Rank, int32& Volume, int32& Dimension, int32& PredOpts)
{
    if (m_ModelHelper->ModelInstance.IsValid())
    {
        // Stop the model if it's running
        if (isModelRunning)
        {
            m_mutex.Lock();
            m_ModelHelper->bIsRunning = false;
            m_mutex.Unlock();
            IsModelRunning = false;
        }

        // Get symbolic output tensor shape
        SymbolicOutputTensorShape = OutputTensorDescs[OutputIdx].GetShape();

        checkf(SymbolicOutputTensorShape.IsConcrete(), TEXT("The current example supports only models without variable input tensor dimensions"));

        OutputTensorShapes = { UE::NNE::FTensorShape::MakeFromSymbolic(SymbolicOutputTensorShape) };
    }
    else
    {
        UE_LOG(LogTemp, Display, TEXT("No Valid Model Instance in Model Helper"));
    }

    Rank = OutputTensorShapes[OutputIdx].Rank();
    Volume = OutputTensorShapes[OutputIdx].Volume();
    Dimension = OutputTensorShapes[OutputIdx].GetData()[0];
    PredOpts = OutputTensorShapes[OutputIdx].GetData()[1];
}


// Sets input tensor shapes in the model instance.
void ANeuralNetwork::SetInputTensorShapes(bool isModelRunning, int32 Rank, int32 Dimension, int32 Frame, int32 ColorChannels, int32 Height, int32 Width)
{
    // MUST BE CALLED ANYTIME INPUT SHAPE CHANGES i.e. image dimensions are different

    TArray<uint32> InputShapeData;

    if (isModelRunning)
    {
        m_mutex.Lock();
        m_ModelHelper->bIsRunning = false;
        m_mutex.Unlock();
        IsModelRunning = false;
    }
    if (Rank == 4)
    {
        InputShapeData.Emplace(Dimension);
        InputShapeData.Emplace(ColorChannels);
        InputShapeData.Emplace(Height);
        InputShapeData.Emplace(Width);
    }
    if (Rank == 5)
    {
        InputShapeData.Emplace(Dimension);
        InputShapeData.Emplace(Frame);
        InputShapeData.Emplace(ColorChannels);
        InputShapeData.Emplace(Height);
        InputShapeData.Emplace(Width);
    }
    if (InputShapeData.Num() == Rank)
    {
        TArray<UE::NNE::FTensorShape> TensorShapes;

        UE::NNE::FTensorShape Shape = UE::NNE::FTensorShape::Make(InputShapeData);
        TensorShapes.Emplace(Shape);
        m_mutex.Lock();
        m_ModelHelper->ModelInstance->SetInputTensorShapes(TensorShapes);
        m_mutex.Unlock();
    }
    else
    {
        m_mutex.Lock();
        m_ModelHelper->ModelInstance->SetInputTensorShapes(InputTensorShapes);
        m_mutex.Unlock();
    }
}


// Creates input tensor bindings for the model instance.
void ANeuralNetwork::CreateInputTensorBinding(bool isModelRunning, bool& InBSuccess)
{
    InBSuccess = false;

    if (isModelRunning)
    {
        m_mutex.Lock();
        m_ModelHelper->bIsRunning = false;
        m_mutex.Unlock();
        IsModelRunning = false;
    }

    // Lock Access
    m_mutex.Lock();

    // Example for creating in- and outputs
    m_ModelHelper->InputData.SetNumZeroed(InputTensorShapes[0].Volume());
    m_ModelHelper->InputBindings.SetNumZeroed(1);
    m_ModelHelper->InputBindings[0].Data = m_ModelHelper->InputData.GetData();
    m_ModelHelper->InputBindings[0].SizeInBytes = m_ModelHelper->InputData.Num() * sizeof(float);

    m_mutex.Unlock();

    InBSuccess = true;

    // Log the new length of InputData after appending
    UE_LOG(LogTemp, Warning, TEXT("Completed! Input Bindings Created: %d"), m_ModelHelper->InputBindings.Num());
}


// Creates output tensor bindings for the model instance.
void ANeuralNetwork::CreateOutputTensorBinding(bool isModelRunning, bool& OutBSuccess)
{
    OutBSuccess = false;

    if (isModelRunning)
    {
        m_mutex.Lock();
        m_ModelHelper->bIsRunning = false;
        m_mutex.Unlock();
        IsModelRunning = false;
    }

    m_mutex.Lock();

    m_ModelHelper->OutputData.SetNumZeroed(OutputTensorShapes[0].Volume());
    m_ModelHelper->OutputBindings.SetNumZeroed(1);
    m_ModelHelper->OutputBindings[0].Data = m_ModelHelper->OutputData.GetData();
    m_ModelHelper->OutputBindings[0].SizeInBytes = m_ModelHelper->OutputData.Num() * sizeof(float);

    m_mutex.Unlock();

    OutBSuccess = true;

    // Log the new length of InputData after appending
    UE_LOG(LogTemp, Warning, TEXT("Completed! Output Bindings Created: %d"), m_ModelHelper->OutputBindings.Num());
}


// Converts UTextureRenderTarget2D to a pixel buffer.
void ANeuralNetwork::RT2PixelBuffer(UTextureRenderTarget2D* InputRT, TArray<FColor>& ImagePixelBuffer, int32& OriginalHeight, int32& OriginalWidth)
{
    OriginalHeight = InputRT->SizeY;
    OriginalWidth = InputRT->SizeX;

    // Check for valid dimensions
    if (OriginalWidth <= 0)
    {
        UE_LOG(LogTemp, Display, TEXT("Image Dimension Error, width is %f"), OriginalWidth)
    }
    if (OriginalHeight <= 0)
    {
        UE_LOG(LogTemp, Display, TEXT("Image Dimension Error, height is %f"), OriginalHeight)
    }

    ImagePixelBuffer.SetNumZeroed(OriginalHeight * OriginalWidth);

    InputRT->GameThread_GetRenderTargetResource()->ReadPixels(ImagePixelBuffer);
}


// Resizes an image using linear interpolation.
void ANeuralNetwork::ResizeImage(TArray<FColor> ImagePixelBuffer, int32 OriginalHeight, int32 OriginalWidth, int32 ResizeHeight, int32 ResizeWidth, TArray<FLinearColor>& ResizedPixelBuffer, int32& ResizedHeight, int32& ResizedWidth)
{
    TArray<FLinearColor> LinearColorArray;
    LinearColorArray.SetNumUninitialized(OriginalWidth * OriginalHeight);

    ParallelFor(OriginalWidth * OriginalHeight, [&](int32 Idx)
        {
            LinearColorArray[Idx] = FLinearColor(ImagePixelBuffer[Idx]);
        });
    if (!ResizedPixelBuffer.IsEmpty())
    {
        ResizedPixelBuffer.Reset();
    }

    ResizedPixelBuffer.SetNumZeroed(ResizeWidth * ResizeHeight);

    FImageUtils::ImageResize(OriginalWidth, OriginalHeight, LinearColorArray, ResizeWidth, ResizeHeight, ResizedPixelBuffer);

    ResizedHeight = ResizeHeight;
    ResizedWidth = ResizeWidth;

    for (FLinearColor colorValue : ResizedPixelBuffer)
    {
        UE_LOG(LogTemp, Error, TEXT("Resized Pixel Buffer, Red: %f, Blue: %f, Green: %f"), colorValue.R, colorValue.G, colorValue.B)
    }
}


// Normalizes an image to a flat array of float values.
void ANeuralNetwork::NormalizeImage(TArray<FLinearColor> ImageBuffer, int32 ColorChannels, TArray<float>& FlatImg)
{
    if (!FlatImg.IsEmpty())
    {
        FlatImg.Reset();
    }
    FlatImg.SetNumZeroed(ImageBuffer.Num() * ColorChannels);
    UE_LOG(LogTemp, Error, TEXT("FlatImg, setsumzeroed length: %d"), FlatImg.Num())

        TArray<float> temp;
    if (!temp.IsEmpty())
    {
        temp.Reset();
    }

    temp.SetNumZeroed(ImageBuffer.Num() * ColorChannels);
    UE_LOG(LogTemp, Error, TEXT("temp, setsumuninitialized length: %d"), temp.Num())


    // Average the RGB Pixel Values to aquire a Grayscale value
    ParallelFor(ImageBuffer.Num(), [&](int32 Idx)
        {
            float GrayValue = FMath::Lerp(ImageBuffer[Idx].R, FMath::Lerp(ImageBuffer[Idx].G, ImageBuffer[Idx].B, 0.33f), 0.33f);

            m_mutex.Lock();
            temp[Idx] = GrayValue;
            m_mutex.Unlock();
        });

    UE_LOG(LogTemp, Error, TEXT("temp, before handoff length: %d"), temp.Num())

        // Copy temp array to FlatImg after the loop
        FlatImg = temp;

    for (float value : FlatImg)
    {
        UE_LOG(LogTemp, Error, TEXT("FlatImg, Value: %f"), value)
    }

    UE_LOG(LogTemp, Error, TEXT("FlatImg, to temp length: %d"), FlatImg.Num())
        temp.Reset();
    UE_LOG(LogTemp, Error, TEXT("temp, after reset length: %d"), temp.Num())
}


// Runs an asynchronous inference using the model instance.
void ANeuralNetwork::RunAsyncInference(bool InBSuccess, bool OutBSuccess, TArray<float> InputData, FNNEAsyncInferenceDelegate Result)
{
    double InferenceStarted = FPlatformTime::Seconds();

    if (m_ModelHelper.IsValid())
    {
        // Example for async inference
        if (!m_ModelHelper->bIsRunning)
        {
            m_mutex.Lock();

            UE_LOG(LogTemp, Error, TEXT("Model is locked"));

            m_ModelHelper->InputData = InputData;
            UE_LOG(LogTemp, Error, TEXT("InputData Length Loaded: %d"), m_ModelHelper->InputData.Num());

            m_ModelHelper->bIsRunning = true;
            IsModelRunning = true;
            UE_LOG(LogTemp, Error, TEXT("Model is running"));

            TSharedPtr<FModelHelper> ModelHelperPtr = m_ModelHelper;
            UE_LOG(LogTemp, Error, TEXT("Model Helper Ptr loaded, beginning async task..."));

            AsyncTask(ENamedThreads::AnyNormalThreadNormalTask, [ModelHelperPtr, Result = MoveTempIfPossible(Result)]()
                {
                    if (ModelHelperPtr->ModelInstance->RunSync(ModelHelperPtr->InputBindings, ModelHelperPtr->OutputBindings) != 0)
                    {
                        UE_LOG(LogTemp, Error, TEXT("Failed to run the model"));
                    }

                    TArray<float> CapturedOutputData = ModelHelperPtr->OutputData;
                    UE_LOG(LogTemp, Error, TEXT("Inference finished, Captured Data Length: %d"), CapturedOutputData.Num());

                    AsyncTask(ENamedThreads::GameThread, [ModelHelperPtr = MoveTempIfPossible(ModelHelperPtr), Result = MoveTempIfPossible(Result), CapturedOutputData = MoveTempIfPossible(CapturedOutputData)]()
                        {
                            UE_LOG(LogTemp, Error, TEXT("Sending to delegate, Captured Data Length: %d"), CapturedOutputData.Num());
                            Result.ExecuteIfBound(CapturedOutputData);
                            ModelHelperPtr->bIsRunning = false;
                            
                            UE_LOG(LogTemp, Error, TEXT("Model is now off, Captured Data Length: %d"), CapturedOutputData.Num());
                            for (float value : CapturedOutputData)
                            {
                                UE_LOG(LogTemp, Warning, TEXT("Prediction Value: %f"), value);
                            }
                        });
                });
            m_mutex.Unlock();
            IsModelRunning = false;
            UE_LOG(LogTemp, Error, TEXT("Model Unlocked"));
        }
        else
        {
            UE_LOG(LogTemp, Error, TEXT("Model is already running"));
        }
    }
    else
    {
        UE_LOG(LogTemp, Error, TEXT("Model helper is not valid"));
    }

    double InferenceStopped = FPlatformTime::Seconds();
    double TimeElapsed = InferenceStopped - InferenceStarted;
    UE_LOG(LogTemp, Warning, TEXT("Inference Time Elapsed: %f"), TimeElapsed);
}


// Callback function for when model data is loaded asynchronously.
void ANeuralNetwork::OnModelDataLoaded()
{
    if (LazyLoadedModelData.IsValid())
    {
        // Log the name of the loaded model
        UE_LOG(LogTemp, Display, TEXT("LazyLoadedModelData loaded %s"), *LazyLoadedModelData.Get()->GetName());

        // Create NNE Runtime
        TWeakInterfacePtr<INNERuntimeCPU> Runtime = UE::NNE::GetRuntime<INNERuntimeCPU>(FString("NNERuntimeORTCpu"));
        if (Runtime.IsValid())
        {
            // Initialize ModelHelper and create the model instance
            m_ModelHelper = MakeShared<FModelHelper>();
            Model = Runtime->CreateModel(LazyLoadedModelData.Get());

            if (Model.IsValid())
            {
                m_mutex.Lock();

                m_ModelHelper->ModelInstance = Model->CreateModelInstance();

                m_mutex.Unlock();

                if (m_ModelHelper->ModelInstance.IsValid())
                {
                    m_mutex.Lock();

                    m_ModelHelper->bIsRunning = false;

                    m_mutex.Unlock();

                    IsModelRunning = false;

                    // Log the success message
                    UE_LOG(LogTemp, Display, TEXT("Model successfully created"));
                }
                else
                {
                    UE_LOG(LogTemp, Display, TEXT("Failed to create Model Instance"));
                }
            }
            else
            {
                UE_LOG(LogTemp, Display, TEXT("Failed to create Model"));
            }
        }
        else
        {
            UE_LOG(LogTemp, Display, TEXT("Failed to create Runtime"));
        }
    }
    else
    {
        UE_LOG(LogTemp, Display, TEXT("LazyLoadedModelData is not valid"));
    }
}


