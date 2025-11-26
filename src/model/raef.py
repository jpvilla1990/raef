import os
import torch
import torch.nn.functional as F
import torch.nn

class RAEF(torch.nn.Module):
    """
    Class to Perform RAEF - Retrieval Augmented Extended Forecasting
    """

    def __init__(
        self
    ):
        """
        Initialize the EmbeddingAugmentation class.

        :param patchSize: Size of the patches to be used in the augmentation process.
        """
        super().__init__()
        self.__device : str = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def extendedAugmentation(self, x : torch.Tensor, context : torch.Tensor, scores : torch.Tensor, thresholdDistance : float) -> torch.Tensor:
        augmented_context = x
        remaining_context = None
        for index in range(context.shape[1]):
            normedScore = scores[0,index] / context.shape[2]
            if normedScore >= thresholdDistance:
                remaining_context = self.__concat(context[0,index,:].unsqueeze(0).unsqueeze(0), remaining_context, dim=1)
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
        scores : torch.Tensor,
        thresholdDistance : float,
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

        xAugmented = self.extendedAugmentation(xNormed, context, scores, thresholdDistance)
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
        self.eval()
        with torch.no_grad():
            output = self.forwardModified(xInput, context, scores)

        return output

    def inferenceExtended(
        self,
        xInput : torch.Tensor,
        context : torch.Tensor,
        scores : torch.Tensor,
        thresholdDistance : float,
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
        self.eval()
        with torch.no_grad():
            output = self.forwardExtended(xInput, context, scores, thresholdDistance)

        return output

    def inference(
        self,
        xInput : torch.Tensor,
        context : torch.Tensor,
        scores : torch.Tensor,
        extended : bool = True,
        thresholdDistance : float = 1.0,
    ):
        if extended:
            return self.inferenceExtended(xInput, context, scores, thresholdDistance)
        else:
            return self.inferenceModified(xInput, context, scores)

if __name__ == "__main__":
    a = RAEF()
    x = torch.randn(5, 31)
    context = torch.randn(5, 3, 33)
    scores = torch.randn(5, 3, 1)
    y = a(x, context, scores)