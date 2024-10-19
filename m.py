import torch
import torch.nn.functional as F  # noqa: N812
from torch.nn import CrossEntropyLoss




class XtrPairwiseCELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.top_k = 3  # Better to use percentage of m(num_query_token)*B(num_document_token) for each batch?
        self.ce_loss = CrossEntropyLoss()

    def forward(self, query_embeddings, doc_embeddings, neg_doc_embeddings):
        """
        query_embeddings: (batch_size, num_query_tokens, dim)
        doc_embeddings: (batch_size, num_doc_tokens, dim)
        Positive scores are the diagonal of the scores matrix.
        """
        print(query_embeddings.shape)
        print(doc_embeddings.shape)
        print(neg_doc_embeddings.shape)
        
        # Compute the ColBERT scores
        # scores = (
        #     torch.einsum("bnd,csd->bcns", query_embeddings, doc_embeddings).max(dim=3)[0].sum(dim=2)
        # )  # (batch_size, batch_size)
        def _score(query_embeddings, doc_embeddings):
            pairwise_scores = torch.einsum(
                "bnd,bsd->bns", query_embeddings, doc_embeddings
            )  # (batch_size, num_query_tokens, num_doc_tokens)

            topk_masks = self.generate_top_k_mask(
                pairwise_scores, self.top_k
            )  # (batch_size, batch_size, num_query_tokens, num_doc_tokens)
            aligned = pairwise_scores * topk_masks
            Z = topk_masks.float().max(-1)[0].sum(-1).clamp(min=1e-3)  # noqa: N806
            doc_tok_summed_normalized = aligned.max(-1)[0].sum(-1) / Z
            scores = doc_tok_summed_normalized  # (batch_size, batch_size)

            # Positive scores are the diagonal of the scores matrix.
            pos_scores = scores.diagonal()  # (batch_size,)
            return pos_scores

        # Negative score for a given query is the maximum of the scores against all all other pages.
        # NOTE: We exclude the diagonal by setting it to a very low value: since we know the maximum score is 1,
        # we can subtract 1 from the diagonal to exclude it from the maximum operation.
        pos_scores = _score(query_embeddings, doc_embeddings)
        neg_scores = _score(query_embeddings, neg_doc_embeddings)

        # Compute the loss
        # The loss is computed as the negative log of the softmax of the positive scores
        # relative to the negative scores.
        # This can be simplified to log-sum-exp of negative scores minus the positive score
        # for numerical stability.
        # torch.vstack((pos_scores, neg_scores)).T.softmax(1)[:, 0].log()*(-1)
        loss = F.softplus(neg_scores - pos_scores).mean()

        return loss

    def generate_top_k_mask(self, tensor, k):
        batch_size, n, s = tensor.shape
        
        # Get the indices of the top k values for each batch
        _, topk_indices = torch.topk(tensor.view(batch_size, -1), k, dim=1)
        
        # Create a mask of zeros with the same shape as the input tensor
        mask = torch.zeros_like(tensor, dtype=torch.bool)
        
        # Convert topk_indices to 3D indices
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, k)
        row_indices = topk_indices // s
        col_indices = topk_indices % s
        
        # Set the top k elements to True in the mask
        mask[batch_indices, row_indices, col_indices] = True
        return mask



def main():
    num_query_tokens = 5
    num_doc_tokens = 10
    dim = 16
    batch_size = 7

    query_embeddings = torch.randn((batch_size, num_query_tokens, dim))
    doc_embeddings = torch.randn((batch_size, num_doc_tokens, dim))
    neg_doc_embeddings = torch.randn((batch_size, num_doc_tokens, dim))
    loss = XtrPairwiseCELoss()
    loss(query_embeddings, doc_embeddings, neg_doc_embeddings)


if __name__ == "__main__":
    main()