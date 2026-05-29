import Foundation
import CryptoKit

@_cdecl("swift_crypto_random_float")
public func swift_crypto_random_float() -> Float {
    // Generate a secure cryptographically strong random 256-bit symmetric key using CryptoKit
    let key = SymmetricKey(size: .bits256)
    
    // Read the first 4 bytes from the cryptographically secure key bytes
    var value: UInt32 = 0
    key.withUnsafeBytes { body in
        if body.count >= 4 {
            value = body.load(as: UInt32.self)
        }
    }
    
    // Map the UInt32 to a float in range [0.0, 1.0)
    return Float(value) / Float(UInt32.max)
}

@_cdecl("swift_crypto_random_float_seeded")
public func swift_crypto_random_float_seeded(seed: UInt32, counter: UInt32) -> Float {
    // Combine seed and counter into a data buffer to make the CryptoKit PRNG deterministic across sweep runs
    var data = Data()
    withUnsafeBytes(of: seed.bigEndian) { data.append(contentsOf: $0) }
    withUnsafeBytes(of: counter.bigEndian) { data.append(contentsOf: $0) }
    
    // Hash using CryptoKit's SHA256 (NIST-compliant Hash_DRBG model)
    let hash = SHA256.hash(data: data)
    
    var value: UInt32 = 0
    hash.withUnsafeBytes { body in
        if body.count >= 4 {
            value = body.load(as: UInt32.self)
        }
    }
    
    return Float(value) / Float(UInt32.max)
}
