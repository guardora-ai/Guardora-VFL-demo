"""
Copyright (с) 2024 Guardora.ai
Non-Commercial Open Software License (NCOSL)
"""

import phe.util
import datetime

from phe import PaillierPublicKey, EncryptedNumber

from utils.log import logger


def serialize_encrypted_number(enc: EncryptedNumber) -> dict:

    if enc.exponent > -32:
        enc = enc.decrease_exponent_to(-32)
        assert enc.exponent == -32
    
    return {
        'v': str(enc.ciphertext()), 
        'e': enc.exponent
    }


def load_encrypted_number(cipher_data, pub_key: PaillierPublicKey) -> EncryptedNumber:

    enc = EncryptedNumber(
        public_key=pub_key, 
        ciphertext=int(cipher_data.v),
        exponent=cipher_data.e
    )

    return enc


def serialize_pub_key(pub_key: PaillierPublicKey) -> str:
    return phe.util.int_to_base64(pub_key.n)


def load_pub_key(pub_key: str) -> PaillierPublicKey:

    n = phe.util.base64_to_int(pub_key)
    pub = phe.PaillierPublicKey(n)

    return pub
    