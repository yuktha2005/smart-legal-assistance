import pytest
from auth.security import get_password_hash, verify_password

def test_password_hashing():
    password = "my_secure_password"
    hashed = get_password_hash(password)
    
    assert hashed != password
    assert verify_password(password, hashed) == True
    assert verify_password("wrong_password", hashed) == False
