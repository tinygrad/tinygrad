import socket

def ipv4_to_gid(ip:str) -> bytes: return bytes(10) + b'\xff\xff' + socket.inet_aton(ip)
