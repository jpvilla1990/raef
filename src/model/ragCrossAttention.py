import os
import torch
import torch.nn.functional as F
import torch.nn

class RagCrossAttention(torch.nn.Module):
    """
    A class to perform cross-attention on embeddings using a multi-head attention mechanism.
    This class is designed to augment embeddings by applying a cross-attention mechanism.
    """

    def __init__(
        self,
        patchSize : int = 16,
        numHeads : int = 8,
        innerDim : int = 256,
        pretrainedModel : str = "",
        loadPretrainedModel : bool = False,
    ):
        """
        Initialize the EmbeddingAugmentation class.

        :param patchSize: Size of the patches to be used in the augmentation process.
        """
        super().__init__()
        self.__device : str = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__patchSize : int = patchSize

        if loadPretrainedModel:
            if os.path.exists(pretrainedModel):
                print(f"Loading pretrained model from {pretrainedModel}")
                self.load_state_dict(
                    torch.load(pretrainedModel, map_location=self.__device, weights_only=False),
                )

    def loadScaleModel(self):
        self.__linealScalerInput : torch.nn.Linear = torch.nn.Linear(self.__patchSize, 256).to(self.__device)
        self.__linealScalerOutput : torch.nn.Linear = torch.nn.Linear(256, 1).to(self.__device)

        self.__reLU : torch.nn.ReLU = torch.nn.ReLU().to(self.__device)

    def loadTwoLinearLayer(self):
        self.__linealInput : torch.nn.Linear = torch.nn.Linear(self.__patchSize, 5112).to(self.__device)
        self.__linealOutput : torch.nn.Linear = torch.nn.Linear(5112, self.__patchSize).to(self.__device)
        self.__reLU : torch.nn.ReLU = torch.nn.ReLU().to(self.__device)

    def loadNoiseModel(self):
        self.__reLU : torch.nn.ReLU = torch.nn.ReLU().to(self.__device)
        self.__linealInputContext : torch.nn.Linear = torch.nn.Linear(self.__patchSize, 512).to(self.__device)
        self.__linealOutputContext : torch.nn.Linear = torch.nn.Linear(512, self.__patchSize).to(self.__device)
        self.__linealInputX : torch.nn.Linear = torch.nn.Linear(self.__patchSize, 512).to(self.__device)
        self.__linealOutputX : torch.nn.Linear = torch.nn.Linear(512, self.__patchSize).to(self.__device)
        self.__sigmoid : torch.nn.Sigmoid = torch.nn.Sigmoid().to(self.__device)

    def __concat(
        self,
        x : torch.Tensor,
        y : torch.Tensor,
        dim : int = -1,
    ) -> torch.Tensor:
        """
        Concatenate two tensors along the last dimension.
        This method is used to concatenate the augmented tensor with the original tensor.
        :param x: First tensor to be concatenated.
        :param y: Second tensor to be concatenated.
        :return: Concatenated tensor.
        """
        if x is None:
            return y
        if y is None:
            return x
        else:
            return torch.cat((x, y), dim=dim)

    def __patching(
        self,
        x : torch.Tensor,
        patchSize : int = None,
    ) -> torch.Tensor:
        """
        Apply patching to the input tensor.
        This method reshapes the input tensor into patches of a specified size.
        :param x: Input tensor to be patched.
        :return: Patches of the input tensor.
        """
        seqLength = x.shape[2]
        patchingSize = self.__patchSize if patchSize is None else patchSize
        remainder : int  = (seqLength - patchingSize) % patchingSize
        padLen : int = patchingSize - remainder if remainder != 0 else 0
        x = F.pad(x, (0, padLen), value=0)
        return x.unfold(dimension=2, size=patchingSize, step=patchingSize)

    def __joinPatches(
        self,
        x : torch.Tensor,
    ) -> torch.Tensor:
        """
        Join patches back to the original tensor shape.
        This method reshapes the patches back to the original tensor shape.
        :param x: Patches to be joined.
        :return: Joined tensor.
        """
        return x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])

    def __normalization(
        self,
        x : torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Normalize the input tensor.
        This method applies layer normalization to the input tensor.
        :param x: Input tensor to be normalized.
        :return: Normalized tensor and its mean and std.
        """
        if len(x.shape) == 3:
            mean : torch.Tensor = x.mean(dim=(1,2), keepdim=True)
            std : torch.Tensor = x.std(dim=(1,2), keepdim=True) + 1e-6
            x = (x - mean) / std
            return x, mean, std
        elif len(x.shape) == 4:
            mean : torch.Tensor = x.mean(dim=(1,2,3), keepdim=True)
            std : torch.Tensor = x.std(dim=(1,2,3), keepdim=True) + 1e-6
            x = (x - mean) / std
            return x, mean, std

    def __denormalization(
        self,
        x : torch.Tensor,
        mean : torch.Tensor,
        std : torch.Tensor,
    ) -> torch.Tensor:
        """
        Denormalize the input tensor.
        This method applies inverse normalization to the input tensor.
        :param x: Input tensor to be denormalized.
        :param mean: Mean used for normalization.
        :param std: Standard deviation used for normalization.
        :return: Denormalized tensor.
        """
        return (x * std) + mean
    
    def __positionalEncoding(
        self,
        x : torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply positional encoding to the input tensor.
        This method adds positional encoding to the input tensor.
        :param x: Input tensor to be augmented with positional encoding.
        :return: Tensor with positional encoding applied.
        """
        dModel : int = self.__patchSize
        nPositions : int = x.shape[2]

        pos : torch.Tensor = torch.arange(
            nPositions,
            dtype=torch.float32,
            device=self.__device,
        ).unsqueeze(1)

        i : torch.Tensor = torch.arange(
            dModel,
            dtype=torch.float32,
            device=self.__device,
        ).unsqueeze(0)

        angleRates : torch.Tensor = 1 / torch.pow(10000, (2 * (i // 2)) / dModel)

        # Apply the formula
        angleRads = pos * angleRates

        # Apply sin to even indices in the array; cos to odd indices
        posEncoding : torch.Tensor = torch.zeros_like(angleRads)
        posEncoding[:, 0::2] = torch.sin(angleRads[:, 0::2])  # even dimensions
        posEncoding[:, 1::2] = torch.cos(angleRads[:, 1::2])  # odd dimensions
        return x + posEncoding.unsqueeze(0).unsqueeze(0)

    def __crossAttention(
        self,
        x : torch.Tensor,
        context : torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply cross-attention to the input tensor.
        This method applies cross-attention to the input tensor using the context tensor.
        :param x: Input tensor to be augmented with cross-attention.
        :param context: Context tensor to be used for cross-attention.
        :return: Tensor after applying cross-attention.
        """
        batchSize = x.shape[0]
        dModel = x.shape[-1]
        query = x.view(batchSize, -1, dModel)
        queryProj : torch.Tensor = self.__linealInput(query)
        queryProj = self.__dropout(queryProj)

        keyValue = context.view(batchSize, -1, dModel)
        keyValueProj : torch.Tensor = self.__linealInput(keyValue)
        keyValueProj = self.__dropout(keyValueProj)

        x = self.__crossAttentionModule(queryProj, keyValueProj, keyValueProj)
        x = self.__dropout(x[0])

        x = self.__linealOutput(x.unsqueeze(1))
        return x

    def gaussian_kernel_1d(self, kernel_size=5, sigma=1.0, device="cpu"):
        """Create a 1D Gaussian kernel."""
        half_size = kernel_size // 2
        x = torch.arange(-half_size, half_size + 1, device=device).float()
        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel /= kernel.sum()
        return kernel.view(1, 1, -1)  # shape: [1, 1, K]

    def smooth(self, context, query):
        """
        smooth function
        """
        context = context.squeeze(1)
        query = query.squeeze(1)
        context_clone = torch.ones_like(context)
        for i in range(context.shape[0]):
            context_i = context[i]
            query_i = query[i]

            diff_context = context_i[1:] - context_i[:-1]
            diff_query = query_i[1:] - query_i[:-1]

            max_diff_context = torch.max(torch.abs(diff_context)) + 1e-6
            max_diff_query = torch.max(torch.abs(diff_query)) + 1e-6
            max_diff = max(max_diff_context, max_diff_query)

            right_context = context_i[-1]
            left_query = query_i[0]
            diff = torch.abs(right_context - left_query)
            steps = int(torch.ceil(diff / max_diff).item()) + int(len(query_i) * 0.1)

            steps_tensor = torch.linspace(right_context, left_query, steps + 3, device=context.device)[1:-2]

            if steps == 0:
                context_clone[i] = context[i].clone()
            else:
                context_clone[i,:-steps] = context[i,steps:].clone()
                context_clone[i,-steps:] = steps_tensor.clone()

        return context_clone.unsqueeze(1)

    def forward2(
        self, xInput : torch.Tensor,
        context : torch.Tensor,
        scores : torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for the embedding augmentation.
        This method applies cross-attention to the input embeddings and returns the augmented embeddings.
        :param x: Input embeddings to be augmented. [batches, seqLength]
        :param context: Context embeddings to be used for augmentation. [batches, k, seqLength]
        :param scores: similarity scores to be used for augmentation. [batches, k]
        :return: Augmented embeddings after applying cross-attention.
        """
        xInput = xInput.to(self.__device)
        context = context.to(self.__device)
        scores = scores.to(self.__device)
        scores = scores.unsqueeze(-1)

        x : torch.Tensor = xInput.unsqueeze(1)

        x = self.__patching(x)
        x, xMean, xStd = self.__normalization(x)
        x = self.__positionalEncoding(x)

        context = self.__patching(context * scores)
        context, _, _ = self.__normalization(context)
        context = self.__positionalEncoding(context)
        x = self.__crossAttention(x, context)

        x = self.__denormalization(x, xMean, xStd)

        x = self.__joinPatches(x)

        xAugmented = x.squeeze(1)

        xAzgmented = self.smooth(xAugmented, xInput)

        xAugmented : torch.Tensor = self.__concat(xAugmented, xInput)

        return xAugmented

    def forwardTwoLinealLayer(
        self, xInput : torch.Tensor,
        context : torch.Tensor,
        scores : torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for the embedding augmentation.
        This method applies cross-attention to the input embeddings and returns the augmented embeddings.
        :param x: Input embeddings to be augmented. [batches, seqLength]
        :param context: Context embeddings to be used for augmentation. [batches, k, seqLength]
        :param scores: similarity scores to be used for augmentation. [batches, k]
        :return: Augmented embeddings after applying cross-attention.
        """
        xInput = xInput.to(self.__device)
        context = context.to(self.__device)
        scores = scores.to(self.__device)

        #scoresTotal : torch.Tensor = scores.sum(dim=1)
        #contextWeighted : torch.Tensor = context * scores.unsqueeze(-1)
        #contextWeighted = contextWeighted.sum(dim=1)

        #contextWeighted = contextWeighted / scoresTotal.unsqueeze(-1)

        x : torch.Tensor = xInput.unsqueeze(1)
        #contextWeighted = contextWeighted.unsqueeze(1)
        context = context.mean(dim=1)
        context = context.unsqueeze(1)

        x = self.__patching(x, self.caPatchSize)
        xNormed, xMean, xStd = self.__normalization(x)

        context = self.__patching(context, self.caPatchSize)
        context = (context - xMean) / xStd

        contextEmbedded : torch.Tensor = self.__linealInput(context)
        xNormedEmbedded : torch.Tensor = self.__linealInput(xNormed)

        contextEmbedded = self.__reLU(contextEmbedded)
        xNormedEmbedded = self.__reLU(xNormedEmbedded)

        contextEmbedded : torch.Tensor = self.__innerLayer(contextEmbedded)
        xNormedEmbedded : torch.Tensor = self.__innerLayer(xNormedEmbedded)

        contextEmbedded = self.__reLU(contextEmbedded)
        xNormedEmbedded = self.__reLU(xNormedEmbedded)

        contextEmbedded = self.__linealOutput(contextEmbedded).squeeze(1).squeeze(-1)
        xNormedEmbedded = self.__linealOutput(xNormedEmbedded).squeeze(1).squeeze(-1)

        augmented : torch.Tensor = contextEmbedded
        augmented[:,:xNormedEmbedded.shape[1]] = augmented[:,:xNormedEmbedded.shape[1]] + xNormedEmbedded
        augmented[:,xNormedEmbedded.shape[1]:] = augmented[:,xNormedEmbedded.shape[1]:] + augmented[:,xNormedEmbedded.shape[1]:]
        augmented = augmented / 2.0
        #augmented = self.__sigmoid(augmented * 4)
        augmented = augmented.unsqueeze(1)

        #augmentedContext = self.__denormalization(context, xMean, xStd)

        augmentedContext = self.__joinPatches(augmented)
        xInput = self.__joinPatches(xNormed)

        augmentedContext = augmentedContext.squeeze(1)
        xInput = xInput.squeeze(1)
        xAugmented : torch.Tensor = self.__concat(augmentedContext, xInput)

        return xAugmented, xMean, xStd

    def forwardScale(
        self, xInput : torch.Tensor,
        context : torch.Tensor,
        scores : torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for the embedding augmentation.
        This method applies cross-attention to the input embeddings and returns the augmented embeddings.
        :param x: Input embeddings to be augmented. [batches, seqLength]
        :param context: Context embeddings to be used for augmentation. [batches, k, seqLength]
        :param scores: similarity scores to be used for augmentation. [batches, k]
        :return: Augmented embeddings after applying cross-attention.
        """
        xInput = xInput.to(self.__device)
        context = context.to(self.__device)
        scores = scores.to(self.__device)

        #scoresTotal : torch.Tensor = scores.sum(dim=1)
        #contextWeighted : torch.Tensor = context * scores.unsqueeze(-1)
        #contextWeighted = contextWeighted.sum(dim=1)

        #contextWeighted = contextWeighted / scoresTotal.unsqueeze(-1)

        x : torch.Tensor = xInput.unsqueeze(1)
        #contextWeighted = contextWeighted.unsqueeze(1)
        context = context.mean(dim=1)
        context = context.unsqueeze(1)

        x = self.__patching(x)
        xNormed, xMean, xStd = self.__normalization(x)

        context = self.__patching(context)
        context = (context - xMean) / xStd

        scaleContext : torch.Tensor = self.__linealScalerInput(context)
        scaleX : torch.Tensor = self.__linealScalerInput(xNormed)

        scaleContext = self.__linealScalerOutput(scaleContext).squeeze(1).squeeze(-1)
        scaleX = self.__linealScalerOutput(scaleX).squeeze(1).squeeze(-1)

        scale : torch.Tensor = scaleContext.sum(1) - scaleX.sum(1)

        scale = self.__reLU(scale)
        #scale = self.__sigmoid(scale)

        #difference : torch.Tensor = x[:,0,0,0] - context[:,-1,-1,-1]
        #context = context + difference.unsqueeze(1).unsqueeze(2).unsqueeze(3)

        context = context + scale.view(scale.shape[0],1,1,1)

        #augmentedContext = self.__denormalization(context, xMean, xStd)

        augmentedContext = self.__joinPatches(context)
        xInput = self.__joinPatches(xNormed)

        augmentedContext = augmentedContext.squeeze(1)
        xInput = xInput.squeeze(1)
        xAugmented : torch.Tensor = self.__concat(augmentedContext, xInput)

        return xAugmented, xMean, xStd

    def forwardNoise(
        self, xInput : torch.Tensor,
        context : torch.Tensor,
        scores : torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for the embedding augmentation.
        This method applies cross-attention to the input embeddings and returns the augmented embeddings.
        :param x: Input embeddings to be augmented. [batches, seqLength]
        :param context: Context embeddings to be used for augmentation. [batches, k, seqLength]
        :param scores: similarity scores to be used for augmentation. [batches, k]
        :return: Augmented embeddings after applying cross-attention.
        """
        xInput = xInput.to(self.__device)
        context = context.to(self.__device)
        scores = scores.to(self.__device)

        x : torch.Tensor = xInput.unsqueeze(1)
        context = context.mean(dim=1)
        context = context.unsqueeze(1)

        x = self.__patching(x)
        xNormed, xMean, xStd = self.__normalization(x)

        context = self.__patching(context)
        context = (context - xMean) / xStd

        contextEmbedded : torch.Tensor = self.__linealInputContext(context)
        xNormedEmbedded : torch.Tensor = self.__linealInputX(xNormed)

        contextEmbedded = self.__reLU(contextEmbedded)
        xNormedEmbedded = self.__reLU(xNormedEmbedded)

        contextEmbedded : torch.Tensor = self.__linealOutputContext(contextEmbedded).squeeze(1).squeeze(-1)
        xNormedEmbedded : torch.Tensor = self.__linealOutputX(xNormedEmbedded).squeeze(1).squeeze(-1)

        noise : torch.Tensor = contextEmbedded
        noise[:,:xNormedEmbedded.shape[1]] = noise[:,:xNormedEmbedded.shape[1]] + xNormedEmbedded
        noise[:,xNormedEmbedded.shape[1]:] = noise[:,xNormedEmbedded.shape[1]:] + noise[:,xNormedEmbedded.shape[1]:]

        noise  = self.__sigmoid(noise * 10)

        noise = noise.unsqueeze(1)
        context = context
        xNormed = xNormed

        noise = self.__joinPatches(noise)
        context = self.__joinPatches(context)
        xInput = self.__joinPatches(xNormed)

        context = context + noise
        xAugmented : torch.Tensor = self.__concat(context, xInput)

        return xAugmented.squeeze(1), xMean, xStd

    def forwardSmoothing(
        self, xInput : torch.Tensor,
        context : torch.Tensor,
        scores : torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for the embedding augmentation.
        This method applies cross-attention to the input embeddings and returns the augmented embeddings.
        :param x: Input embeddings to be augmented. [batches, seqLength]
        :param context: Context embeddings to be used for augmentation. [batches, k, seqLength]
        :param scores: similarity scores to be used for augmentation. [batches, k]
        :return: Augmented embeddings after applying cross-attention.
        """
        xInput = xInput.to(self.__device)
        context = context.to(self.__device)
        scores = scores.to(self.__device)
        x : torch.Tensor = xInput.unsqueeze(1)

        xNormed, xMean, xStd = self.__normalization(x)
        context = context.mean(dim=1)
        context = (context - xMean) / xStd

        context = self.smooth(context, xNormed)

        xAugmented : torch.Tensor = self.__concat(context, xNormed)

        return xAugmented.squeeze(1), xMean, xStd

    def similarityWindow(self, x : torch.Tensor, context : torch.Tensor, window_size : int = 16, threshold : float = 0.2) -> torch.Tensor:
        """
        Apply a similarity-based windowing mechanism to the context embeddings.
        This method selects a window of context embeddings based on their similarity to the input embeddings.
        :param x: Input tensor to be augmented. [batches, 1, seqLength]
        :param context: Context tensor to be used for augmentation. [batches, 1, seqLength]
        :param window_size: Size of the window to be selected.
        :return: Tensor after applying the similarity-based windowing mechanism.
        """
        x = x.view(-1)
        context = context.view(-1)
        context_interest = context[:-self.__patchSize]

        n = x.shape[0]
        kept_indices = n - window_size  # start from end - window, move backward

        for start in range(n - window_size, -1, -window_size):
            x_win = x[start:start + window_size]
            context_win = context_interest[start:start + window_size]

            # Euclidean distance between the two windows
            dist = torch.norm(x_win - context_win, p=1).item() / window_size

            if dist > threshold:
                break
            else:
                # keep going, include this window
                kept_indices = start

        # append only the portion of b we kept
        return context[kept_indices:].unsqueeze(0).unsqueeze(0)

    def extendedAugmentation(self, x : torch.Tensor, context : torch.Tensor, scores : torch.Tensor) -> torch.Tensor:
        augmented_context = x
        remaining_context = None
        maxScore = 2.0
        for index in range(context.shape[1]):
            normedScore = scores[0,index] / context.shape[2]
            if normedScore >= 1.0 and normedScore < maxScore:
                remaining_context = self.__concat(context[0,index,:].unsqueeze(0).unsqueeze(0), remaining_context, dim=1)
            elif normedScore >= maxScore:
                continue
            else:
                augmented_context = self.__concat(context[0,index,:].unsqueeze(0).unsqueeze(0), augmented_context)

        if remaining_context is not None:
            remaining_context = remaining_context.mean(dim=1).unsqueeze(0)
            augmented_context = self.__concat(remaining_context, augmented_context)

        return augmented_context

    def forwardModified(
        self, xInput : torch.Tensor,
        context : torch.Tensor,
        scores : torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for the embedding augmentation.
        This method applies cross-attention to the input embeddings and returns the augmented embeddings.
        :param x: Input embeddings to be augmented. [batches, seqLength]
        :param context: Context embeddings to be used for augmentation. [batches, k, seqLength]
        :param scores: similarity scores to be used for augmentation. [batches, k]
        :return: Augmented embeddings after applying cross-attention.
        """
        xInput = xInput.to(self.__device)
        context = context.to(self.__device)
        scores = scores.to(self.__device)
        x : torch.Tensor = xInput.unsqueeze(1)

        xNormed, xMean, xStd = self.__normalization(x)
        context = context.mean(dim=1)
        context = (context - xMean) / xStd

        xAugmented : torch.Tensor = self.__concat(context, xNormed)
        return xAugmented.squeeze(1), xMean, xStd

    def forwardExtended(
        self, xInput : torch.Tensor,
        context : torch.Tensor,
        scores : torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for the embedding augmentation.
        This method applies cross-attention to the input embeddings and returns the augmented embeddings.
        :param x: Input embeddings to be augmented. [batches, seqLength]
        :param context: Context embeddings to be used for augmentation. [batches, k, seqLength]
        :param scores: similarity scores to be used for augmentation. [batches, k]
        :return: Augmented embeddings after applying cross-attention.
        """
        xInput = xInput.to(self.__device)
        context = context.to(self.__device)
        scores = scores.to(self.__device)
        x : torch.Tensor = xInput.unsqueeze(1)

        xNormed, xMean, xStd = self.__normalization(x)
        context = (context - xMean) / xStd
        scores = scores / (xStd.squeeze() ** 2)

        xAugmented = self.extendedAugmentation(xNormed, context, scores)
        return xAugmented.squeeze(1), xMean, xStd

    def inferenceModified(
        self,
        xInput : torch.Tensor,
        context : torch.Tensor,
        scores : torch.Tensor
    ) -> torch.Tensor:
        """
        Inference pass for the embedding augmentation.
        This method applies cross-attention to the input embeddings and returns the augmented embeddings.
        :param x: Input embeddings to be augmented. [batches, seqLength]
        :param context: Context embeddings to be used for augmentation. [batches, k, seqLength]
        :param scores: similarity scores to be used for augmentation. [batches, k, 1]
        :return: Augmented embeddings after applying cross-attention.
        """
        output : torch.Tensor = None
        #self.eval()
        #with torch.no_grad():
        output = self.forwardModified(xInput, context, scores)

        return output

    def inferenceExtended(
        self,
        xInput : torch.Tensor,
        context : torch.Tensor,
        scores : torch.Tensor
    ) -> torch.Tensor:
        """
        Inference pass for the embedding augmentation.
        This method applies cross-attention to the input embeddings and returns the augmented embeddings.
        :param x: Input embeddings to be augmented. [batches, seqLength]
        :param context: Context embeddings to be used for augmentation. [batches, k, seqLength]
        :param scores: similarity scores to be used for augmentation. [batches, k, 1]
        :return: Augmented embeddings after applying cross-attention.
        """
        output : torch.Tensor = None
        #self.eval()
        #with torch.no_grad():
        output = self.forwardExtended(xInput, context, scores)

        return output

    def inference(
        self,
        xInput : torch.Tensor,
        context : torch.Tensor,
        scores : torch.Tensor,
        extended : bool,
    ):
        if extended:
            return self.inferenceExtended(xInput, context, scores)
        else:
            return self.inferenceModified(xInput, context, scores)

if __name__ == "__main__":
    a = RagCrossAttention()
    x = torch.randn(5, 31)
    context = torch.randn(5, 3, 33)
    scores = torch.randn(5, 3, 1)
    y = a(x, context, scores)