import torch
from loguru import logger
from tqdm import tqdm


def check_byte_mappings(tokenizer):
    """
    Args:
    - tokenizer: An object representing a tokenizer. This tokenizer object must have a method
                 `get_vocab()` that returns a dictionary mapping tokens to their respective
                 token IDs within the tokenizer's vocabulary.

    Returns:
    - If the tokenizer is identified as BBPE based on prefix counts, returns a dictionary for byte values from '<0x00>' to '<0x7F>'.
    - Otherwise, returns a byte_mapping (dict): A dictionary where each key is a string representing a byte value in
                           standard hex format (e.g., '<0x00>', '<0x01>', ..., '<0xFF>'), and each
                           value is the corresponding token ID for that byte representation
                           within the tokenizer's vocabulary.
    """
    vocab = tokenizer.get_vocab()
    g_prefix_count = sum(token.startswith("Ġ") for token in vocab)
    u_prefix_count = sum(token.startswith("▁") for token in vocab)

    byte_mapping = {}

    # For BBPE, handle bytes from 0x00 to 0x7F
    if g_prefix_count > u_prefix_count:
        for byte_val in range(128):  # Limit to 0x00 to 0x7F
            byte_char = chr(byte_val)
            token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(byte_char))[0]
            hex_token = f"<0x{byte_val:02X}>"
            byte_mapping[hex_token] = token_id
    else:
        # For non-BBPE, attempt to find a direct mapping in vocab
        for byte_val in range(256):
            hex_token = f"<0x{byte_val:02X}>"
            # For cases like "\t" being replaced in vocab
            if hex_token == "<0x09>" and hex_token not in vocab:
                continue
            if hex_token not in vocab:
                raise ValueError(
                    f"Token {hex_token} not found in tokenizer's vocabulary."
                )
            byte_mapping[hex_token] = vocab[hex_token]

    return byte_mapping


def get_vocab_union_and_mapping(tokenizers):
    """
    Modified function that creates a union of tokens from the vocabularies of given tokenizers and
    provides a mapping for each tokenizer from its token IDs to the tokens in the unified vocabulary.
    It handles tokens starting with 'Ġ' or '▁' differently to merge similar tokens.

    Args:
    tokenizers (list): A list of tokenizer objects, each with a 'get_vocab()' method that
                       returns a dictionary of tokens and their corresponding IDs in the tokenizer's
                       vocabulary.

    Returns:
    tuple: A tuple containing three elements:
        - vocab_union (set): A set containing the union of all tokens in the vocabularies of the
                             provided tokenizers.
        - tokenizers_mapping (list): A list of dictionaries, where each dictionary corresponds to
                                     a tokenizer from the input list and maps token IDs from the
                                     tokenizer to tokens in the vocab_union.
        - index_to_vocab (dict): A dictionary mapping from unique index to tokens in the vocab_union.
        - byte_mappings_list (list): A list of dictionaries, where each dictionary corresponds to a
                                tokenizer from the input list and provides a mapping of byte value
                                tokens from '<0x00>' to '<0xFF>' to their original token IDs in the
                                tokenizer's vocabulary. This mapping is used to ensure consistency
                                and to facilitate the identification and replacement of these tokens
                                in the unified vocabulary.
    """
    # Initialize a set to store all tokens
    vocab_union = set()
    # Initialize a list to store the mappings for each tokenizer
    tokenizers_mapping = []
    byte_mappings_list = []

    # First, add '<0x00>' to '<0xFF>'
    for byte_val in range(256):
        vocab_union.add(f"<0x{byte_val:02X}>")

    # Process each tokenizer separately
    for tokenizer in tokenizers:
        vocab = tokenizer.get_vocab()
        token_set = set()
        mapping = {}

        # Check and record each tokenizer's mapping for '<0x00>' to '<0xFF>'
        byte_mapping = check_byte_mappings(tokenizer)
        byte_mappings_list.append(byte_mapping)

        if len(byte_mapping) == 128:
            logger.warning(
                "BBPE detected. Please be cautious in usage as currently it only supports applications such as multiple-choice questions eg.(A)"
            )

        # Remove the existing mappings for '<0x00>' to '<0xFF>'
        for hex_token, token_id in byte_mapping.items():
            # Remove tokens from the vocabulary whose token IDs appear in the byte_mapping
            actual_tokens = [token for token, id in vocab.items() if id == token_id]

            if len(actual_tokens) != 1:
                # Raise an error if more than one matching token is found
                raise ValueError(
                    f"Multiple tokens/ Zero token found for token ID {token_id} in tokenizer's vocabulary."
                )
            del vocab[actual_tokens[0]]

        # Detect usage of 'Ġ' and '▁'
        g_prefix_count = sum(token.startswith("Ġ") for token in vocab)
        u_prefix_count = sum(token.startswith("▁") for token in vocab)

        # Process tokens based on prefix type
        if g_prefix_count > u_prefix_count:
            # Handle tokens starting with 'Ġ'
            for token, token_id in vocab.items():
                processed_token = token.replace("Ġ", " ").replace("Ċ", "\n")
                token_set.add(processed_token)
                mapping[token_id] = processed_token
        else:
            # Handle tokens starting with '▁'
            for token, token_id in vocab.items():
                if token.startswith("▁"):
                    processed_token = token.replace("▁", " ")
                else:
                    # For tokens without '▁', use the decode method
                    processed_token = token  # tokenizer.decode([token_id])
                token_set.add(processed_token)
                mapping[token_id] = processed_token

        # Merge into the total vocab_union
        vocab_union = vocab_union.union(token_set)
        # Append the mapping for this tokenizer to the list
        tokenizers_mapping.append(mapping)

    # Generate a mapping for each token in the union to a unique index
    vocab_to_index = {token: i for i, token in enumerate(vocab_union)}

    # Convert vocab_to_index to index_to_vocab
    index_to_vocab = {index: token for token, index in vocab_to_index.items()}

    for tokenizer, byte_mapping, mapping in zip(
        tokenizers, byte_mappings_list, tokenizers_mapping
    ):
        # Update the mappings for each tokenizer to map to the index in the unified vocab
        for token_id, token in mapping.items():
            mapping[token_id] = vocab_to_index[token]

        # Define the extended mapping dictionary
        bbpe_mapping = {
            **{
                f"<0x{hex(i)[2:].upper()}>": chr(i) for i in range(0x30, 0x3A)
            },  # mapping '0' to '9'
            **{
                f"<0x{hex(i)[2:].upper()}>": chr(i) for i in range(0x41, 0x5B)
            },  # mapping 'A' to 'Z'
            **{
                f"<0x{hex(i)[2:].upper()}>": chr(i) for i in range(0x61, 0x7B)
            },  # mapping 'a' to 'z'
        }

        # Add the '<0x00>' to '<0xFF>' mappings for each tokenizer
        for hex_token, original_token_id in byte_mapping.items():
            # First, check the original conditions
            if (
                not all(len(bm) == 128 for bm in byte_mappings_list)
                and len(byte_mapping) == 128
            ):
                # Apply special handling to the specified characters
                if hex_token in bbpe_mapping:
                    logger.warning(
                        f"We force-mapped the BBPE {hex_token} to {bbpe_mapping[hex_token]} in union vocab"
                    )
                    mapping[original_token_id] = vocab_to_index[bbpe_mapping[hex_token]]
                    continue
            mapping[original_token_id] = vocab_to_index[hex_token]

    return vocab_union, tokenizers_mapping, index_to_vocab, byte_mappings_list


def create_mapping_matrix(mapping, union_vocab_size, model_vocab_size):
    """
    Creates a sparse tensor mapping matrix for vocabulary translation.

    Args:
    - mapping (dict): Maps model token IDs to unified vocabulary indexes.
    - union_vocab_size (int): Size of the unified vocabulary.
    - model_vocab_size (int): Size of the model's vocabulary.

    Returns:
    - torch.sparse_coo_tensor: Sparse tensor in COO format with shape [model_vocab_size, union_vocab_size].
                               Each non-zero element (i, j) indicates a mapping from the i-th token in the
                               model's vocabulary to the j-th token in the unified vocabulary.
    """

    indices = []  # Store the coordinates of non-zero elements
    values = []  # Non-zero values, typically 1 for a mapping matrix

    for model_token_id, unified_token_index in mapping.items():
        if model_token_id < model_vocab_size:
            indices.append([model_token_id, unified_token_index])  # (rows, cols)
            values.append(1.0)

    # Convert to a tensor suitable for COO format
    indices = torch.tensor(
        indices, dtype=torch.long
    ).t()  # Transpose to meet (rows, cols)
    values = torch.tensor(values, dtype=torch.float)

    # Create a sparse tensor
    size = torch.Size([model_vocab_size, union_vocab_size])
    mapping_matrix = torch.sparse_coo_tensor(indices, values, size)

    return mapping_matrix


def get_token_ids(tokenizer, token, special_prefix_token, byte_mapping):
    """
    Tokenizes a given token and a special prefix token from the tokenizer's vocabulary,
    then finds the token IDs for the portion of the given token that does not overlap
    with the special prefix token. It is particularly useful for identifying unique sub-tokens
    in tokenization processes. If initial tokenization does not meet expectations,
    it tries using ';' as an alternate special prefix token.

    Args:
    tokenizer: An instance of a tokenizer class with an 'encode' method that converts
               text to a list of token IDs.
    token (str): The token to be tokenized and analyzed.
    special_prefix_token (str): A special prefix token from the tokenizer's vocabulary, used as a
                                reference point for comparison. It is the shortest token starting with
                                a specific prefix ('▁' in most cases), which is neither part of any
                                other token nor contains any other token.
    byte_mapping (dict): A dictionary mapping standard byte representations ('<0x00>' to '<0xFF>')
                         to their token IDs in the tokenizer's vocabulary.

    Returns:
    list: A list of token IDs representing the non-overlapping part of the 'token'
          when tokenized, compared to the tokenization of 'special_prefix_token'.

    The function tries using the provided special_prefix_token, and if tokenization doesn't match as expected,
    it attempts using ';' as an alternate special_prefix_token. If it still doesn't match, it returns
    the token IDs for 'token'.
    """

    # Check if the token is a standard byte representation and return its token ID if found
    if token in byte_mapping:
        return [byte_mapping[token]]

    if byte_mapping != 128:
        prefix_tokens = [special_prefix_token, ";"]

        for prefix_token in prefix_tokens:
            # Tokenize individually
            token_id_list1 = tokenizer.encode(prefix_token, add_special_tokens=False)

            # Tokenize doubled token
            token_id_list2 = tokenizer.encode(
                prefix_token + token, add_special_tokens=False
            )

            # Check if the start of token_id_list2 matches token_id_list1
            if token_id_list2[: len(token_id_list1)] == token_id_list1:
                result = token_id_list2[len(token_id_list1) :]
                if result:
                    return result

        # If tokenization doesn't match as expected with any prefix token, return the token IDs for 'token'
        logger.warning(f"Warning: Token '{token}' may not be tokenized as expected.")
    return tokenizer.encode(token, add_special_tokens=False)


def find_special_underscore_token(tokenizer):
    """
    Identifies the shortest special token in the tokenizer's vocabulary that starts with '▁',
    which is neither part of any other token nor contains any other token (except '▁' itself).
    '▁' itself and tokens resulting in only whitespace after '▁' is removed are also excluded
    from the result.

    Args:
        tokenizer: An instance of a tokenizer class with a 'get_vocab()' method, returning
                   a dictionary of tokens and their IDs.

    Returns:
        str: The shortest special token meeting the criteria, with '▁' removed, sorted
             lexicographically to ensure consistency. Raises an error if no such token is found.

    The function first checks the prevalence of tokens starting with 'Ġ' and '▁'. If tokens
    starting with 'Ġ' are more prevalent, it returns an empty string. Otherwise, it proceeds
    to find the shortest token starting with '▁', which is not part of any other token and
    does not contain any other tokens (except for the initial '▁'), and is not just whitespace
    after '▁' is removed. It then removes '▁' from the token before returning it. If no such
    token is found, an error is raised.
    """

    # get tokenizer vocab
    vocab = tokenizer.get_vocab()

    # Count tokens that start with 'Ġ' and '▁'
    count_prefix_G = sum(1 for token in vocab if token.startswith("Ġ"))
    count_prefix_underscore = sum(1 for token in vocab if token.startswith("▁"))

    # Return an empty string if 'Ġ' tokens are more frequent
    if count_prefix_G > count_prefix_underscore:
        return ""

    # Filter tokens that start with '▁'
    underscore_tokens = [
        token for token in vocab if token.startswith("▁") and token != "▁"
    ]

    # Filter tokens that meet the criteria
    special_tokens = []
    for token in tqdm(underscore_tokens, desc="Analyzing tokens"):
        cleaned_token = token[1:]  # remove '▁'

        # Ensure the token is not part of another token, contains no additional tokens besides the first '▁',
        # has no multiple '▁', and is not a space after removing '▁'
        if (
            not any(
                token in other_token
                for other_token in underscore_tokens
                if other_token != token
            )
            and token.count("▁") == 1
            and cleaned_token.strip() != ""
        ):
            special_tokens.append(cleaned_token)

    # Raise an error if no token meets the criteria
    if not special_tokens:
        raise ValueError("No special underscore token found that meets the criteria.")

    # Return the shortest token to ensure consistency
    return min(special_tokens, key=lambda x: (len(x), x))


def get_special_prefix_tokens_for_all(tokenizers):
    """
    This function takes a list of tokenizers and returns a dictionary where each tokenizer is
    associated with its special prefix token. It utilizes a hypothetical function find_special_underscore_token
    which is assumed to return the special prefix token that each individual tokenizer can handle.

    Args:
    tokenizers (list): A list of tokenizer objects. Each tokenizer is assumed to have a
                       method or functionality that allows the extraction of its special prefix token.

    Returns:
    dict: A dictionary where each key is a tokenizer from the input list, and the corresponding
          value is the special prefix token that the tokenizer can handle, as determined by calling
          the find_special_underscore_token function.

    Example:
    tokenizers = [tokenizer1, tokenizer2, ...]
    special_prefix_tokens = get_special_prefix_tokens_for_all(tokenizers)
    print(special_prefix_tokens)  # Output: {tokenizer1: special_prefix_token1, tokenizer2: special_prefix_token2, ...}
    """

    # Initialize an empty dictionary to store the results
    special_prefix_tokens = {}

    # Iterate through the list of tokenizers
    for tokenizer in tokenizers:
        # Get the special prefix token for each tokenizer
        token = find_special_underscore_token(tokenizer)
        # Store the tokenizer and its special prefix token in the dictionary
        special_prefix_tokens[tokenizer] = token
    return special_prefix_tokens


class TokenMapper:
    def __init__(self, tokenizer_infos):
        from transformers import AutoTokenizer # type: ignore

        self.infos = tokenizer_infos
        self.same_tokenizer = all(
            info["vocab"] == tokenizer_infos[0]["vocab"] for info in tokenizer_infos
        )
        if self.same_tokenizer:
            # Identical tokenizers ensemble directly in the original vocab.
            return

        self.tokenizers = [
            AutoTokenizer.from_pretrained(info["path"], trust_remote_code=True)
            for info in tokenizer_infos
        ]
        self.special_prefix_tokens_dict = get_special_prefix_tokens_for_all(self.tokenizers)
        self.prefix_tokens = [
            self.special_prefix_tokens_dict[tokenizer] for tokenizer in self.tokenizers
        ]
        (
            self.vocab_union,
            self.token_id_to_union,
            index_to_vocab,
            self.byte_mappings,
        ) = get_vocab_union_and_mapping(self.tokenizers)
        self.index_to_token = [
            index_to_vocab[index] for index in range(len(index_to_vocab))
        ]
        self.union_to_token_ids = []
        self._encoded_union_cache = {}
        for mapping, info in zip(self.token_id_to_union, tokenizer_infos):
            union_to_token_ids = {}
            for token_id, union_id in mapping.items():
                if token_id < info["vocab_size"]:
                    union_to_token_ids.setdefault(union_id, []).append(token_id)
            self.union_to_token_ids.append(union_to_token_ids)
        self.matrices = [
            create_mapping_matrix(mapping, len(self.vocab_union), info["vocab_size"])
            for mapping, info in zip(self.token_id_to_union, tokenizer_infos)
        ]

    def _fit_probs(self, probs, model_idx):
        vocab_size = self.matrices[model_idx].shape[0]
        if probs.numel() == vocab_size:
            return probs.float()
        if probs.numel() > vocab_size:
            return probs[:vocab_size].float()
        padded = torch.zeros(vocab_size, dtype=torch.float32)
        padded[: probs.numel()] = probs.float()
        return padded

    def merge_probs(self, indexed_probs, weights):
        merged = torch.zeros(len(self.index_to_token), dtype=torch.float32)
        for model_idx, probs in indexed_probs:
            model_probs = self._fit_probs(probs, model_idx).unsqueeze(0)
            mapped = torch.sparse.mm(model_probs, self.matrices[model_idx]).squeeze(0)
            merged += weights[model_idx] * mapped
        return merged / merged.sum()

    def merge_topk_probs(self, indexed_probs, weights, top_k):
        fitted = [
            (model_idx, self._fit_probs(probs, model_idx))
            for model_idx, probs in indexed_probs
        ]
        union_ids = set()
        for model_idx, probs in fitted:
            for token_id in torch.topk(probs, min(top_k, probs.numel())).indices.tolist():
                union_id = self.token_id_to_union[model_idx].get(token_id)
                if union_id is not None:
                    union_ids.add(union_id)

        union_ids = sorted(union_ids)
        merged = torch.zeros(len(self.index_to_token), dtype=torch.float32)
        for model_idx, probs in fitted:
            values = torch.stack(
                [self._union_prob(probs, model_idx, union_id) for union_id in union_ids]
            )
            values = values / values.sum()
            for union_id, prob in zip(union_ids, values):
                merged[union_id] += weights[model_idx] * prob
        return merged

    def _union_prob(self, probs, model_idx, union_id):
        token_ids = self.union_to_token_ids[model_idx].get(union_id)
        if token_ids:
            return probs[token_ids].sum()

        key = (model_idx, union_id)
        if key not in self._encoded_union_cache:
            token = self.index_to_token[union_id]
            info = self.infos[model_idx]
            self._encoded_union_cache[key] = self._encode_token(
                model_idx, info, self.prefix_tokens[model_idx], token
            )
        # UniTE scores an out-of-vocabulary union token by its first subtoken.
        return probs[self._encoded_union_cache[key][0]]

    def convert_model_token(self, model_idx, token_id):
        if self.same_tokenizer:
            return [[token_id] for _ in self.infos]

        union_id = self.token_id_to_union[model_idx].get(token_id)
        if union_id is None:
            token = next(
                token
                for token, idx in self.infos[model_idx]["vocab"].items()
                if idx == token_id
            )
            union_id = self.index_to_token.index(token)
        return self.convert_union_token(union_id)

    def convert_union_token(self, union_id):
        token = self.index_to_token[union_id]
        return [
            self._encode_token(idx, info, prefix, token)
            for idx, (info, prefix) in enumerate(zip(self.infos, self.prefix_tokens))
        ]

    def _encode_token(self, idx, info, prefix, token):
        if token == info["eos_token"] or token in {
            "<|end_of_text|>",
            "<|endoftext|>",
            "<|im_end|>",
            "<|end|>",
        }:
            return [info["eos_token_id"]]
        return get_token_ids(
            self.tokenizers[idx], token, prefix, self.byte_mappings[idx]
        )
