"""
Author: Moises Badajoz Martinez <m.badajozmartinez@ugto.mx>

University of Guanajuato (2025)
"""

import serial
from dataclasses import dataclass
import struct


def unpacket_data_to_dict(data: tuple) -> dict:
    """Desempaqueta los una tupla de datos de sensores en un diccionario

    Args:
        data (tuple): tupla de datos de sensores

    Returns:
        dict: diccionario con los datos organizados
    """
    dict_data = {
        "type": data[0],
        "timestamp": data[2],
        "sensor_count": data[1],
        "data": {
            "".join([format(x, "02X") for x in data[3 + (i * 16) : 9 + (i * 16)]]): {
                "quat": list(data[9 + (i * 16) : 13 + (i * 16)]),
                "accel": list(data[13 + (i * 16) : 16 + (i * 16)]),
                "gyro": list(data[16 + (i * 16) : 19 + (i * 16)]),
            }
            for i in range(data[1])
        },
    }
    return dict_data


def unpacket_bat_to_dict(data: tuple) -> dict:
    """Desempaqueta los una tupla de datos de bateria de sensores en un diccionario

    Args:
        data (tuple): tupla de datos de bateria de sensores

    Returns:
        dict: diccionario con los datos organizados
    """
    dict_bat = {
        "type": data[0],
        "sensor_count": data[1],
        "data": {
            "".join([format(x, "02X") for x in data[2 + (i * 7) : 8 + (i * 7)]]): data[
                8 + (i * 7)
            ]
            for i in range(data[1])
        },
    }
    return dict_bat


def unpack(buffer: bytearray, FORMAT: str) -> tuple:
    """Desempaqueta un buffer de bytes en una tupla de datos dado un formato
    struct.unpack() hace lo mismo pero aqui verifica que la longitud del buffer sea suficiente,
    si no, devuelve una tupla vacia

    Args:
        buffer (bytearray): buffer de bytes
        FORMAT (str): formato para desempaquetar, revisar documentacion del modulo struct

    Returns:
        tuple: tupla de datos desempaquetados
    """

    EXPECTED_BYTES = struct.calcsize(FORMAT)
    if len(buffer) < EXPECTED_BYTES:
        return ()
    return struct.unpack(FORMAT, buffer[:EXPECTED_BYTES])


def unpack_data(buffer: bytearray, FORMAT: str) -> dict:
    """desempaqueta un buffer de bytes en un diccionario dado un formato
    * para datos de sensores

    Args:
        buffer (bytearray): buffer de bytes
        FORMAT (str): formato para desempaquetar, revisar documentacion del modulo struct

    Returns:
        dict: diccionario con la informacion ordenada
    """
    data = unpack(buffer, FORMAT)
    if data == ():
        return {}
    return unpacket_data_to_dict(data)


def unpack_bat(buffer: bytearray, FORMAT: str) -> dict:
    """desempaqueta un buffer de bytes en un diccionario dado un formato
    * para informacion de bateria de sensores
    Args:
        buffer (bytearray): buffer de bytes
        FORMAT (str): formato para desempaquetar, revisar documentacion del modulo struct

    Returns:
        dict: diccionario con la informacion ordenada
    """
    data = unpack(buffer, FORMAT)
    if data == ():
        return {}
    return unpacket_bat_to_dict(data)


@dataclass
class McdAPI:
    """Clase para la comunicacion con el dispositivo"""

    __port: str  # puerto serial
    __baudrate: int  # baudrate del puerto

    def set_port(self, port: str) -> None:
        """Asigna el puerto

        Args:
            port (str): puerto serial
        """
        self.__port = port

    def set_baudrate(self, baudrate: int) -> None:
        """Asigna el baudrate

        Args:
            baudrate (int): velocidad de baudios
        """
        self.__baudrate = baudrate

    def set_config(self, port: str, baudrate: int) -> None:
        """Asigna el puerto y el baudrate

        Args:
            port (str): puerto serial
            baudrate (int): velocidad de baudios
        """
        self.set_port(port)
        self.set_baudrate(baudrate)

    def connect(self) -> bool:
        """Conecta con el puerto serial

        Returns:
            bool: si conecto o no
        """
        try:
            self.__ser = serial.Serial(self.__port, self.__baudrate)
            return self.__ser.is_open
        except Exception as e:
            print(e)
            return False

    def close(self) -> bool:
        """Cierra la conexion con el puerto serial

        Returns:
            bool: si se desconecto o no
        """
        try:
            self.read_stop()
            self.__ser.close()
            return not self.__ser.is_open
        except OSError as e:
            return True
        except:
            return False

    def is_open(self) -> bool:
        """Retorna si esta o no abierta la conexion con el puerto serial

        Returns:
            bool: estado de la conexion
        """
        try:
            return self.__ser.is_open
        except:
            return False

    def read(self) -> bytes | None:
        """Retorna la lectura del puerto serial

        Returns:
            (bytes | None): buffer de bytes
        """
        return self.__ser.read_all()

    def in_waiting(self) -> int:
        """Retorna la cantidad de bytes en espera de ser leidos

        Returns:
            int: contidad de bytes en el puerto serial
        """
        return self.__ser.in_waiting

    def sync_mode(self) -> None:
        """Envia el mensaje para entrar en modo sincronizacion"""
        buffer = b"\x03"
        self.__ser.write(buffer)

    def identify_sensor(self, addr: str = "FFFFFFFFFFFF") -> None:
        """Envia el mensaje para identificar en uno o todos los sensores

        Args:
            addr (str, optional): Direccion objetivo. Defaults to 'FFFFFFFFFFFF'.
        """
        mac = bytes.fromhex(addr)
        buffer = b"\x07" + mac
        self.__ser.write(buffer)

    def read_start(self, addr: str = "FFFFFFFFFFFF") -> None:
        """Envia el mensaje para iniciar la lectura en uno o todos los sensores

        Args:
            addr (str, optional): Direccion objetivo. Defaults to 'FFFFFFFFFFFF'.
        """
        mac = bytes.fromhex(addr)
        buffer = b"\x08" + mac
        self.__ser.write(buffer)

    def read_stop(self, addr: str = "FFFFFFFFFFFF") -> None:
        """Envia el mensaje para detener la lectura en uno o todos los sensores

        Args:
            addr (str, optional): Direccion objetivo. Defaults to 'FFFFFFFFFFFF'.
        """
        mac = bytes.fromhex(addr)
        buffer = b"\x09" + mac
        self.__ser.write(buffer)

    def reset_sensor(self, addr: str = "FFFFFFFFFFFF") -> None:
        """Envia el mensaje para resetear uno o todos los sensores

        Args:
            addr (str, optional): Direccion objetivo. Defaults to 'FFFFFFFFFFFF'.
        """
        mac = bytes.fromhex(addr)
        buffer = b"\x0a" + mac
        self.__ser.write(buffer)

    def calibrate_sensor(self, addr: str = "FFFFFFFFFFFF") -> None:
        """Envia el mensaje para calibrar uno o todos los sensores

        Args:
            addr (str, optional): Direccion objetivo. Defaults to 'FFFFFFFFFFFF'.
        """
        mac = bytes.fromhex(addr)
        buffer = b"\x0b" + mac
        self.__ser.write(buffer)
