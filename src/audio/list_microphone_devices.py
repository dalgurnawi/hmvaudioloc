import sounddevice as sd


def list_microphone_devices():
    """
    Execute to get a list of current input/output devices.
    Choose appropriate int identifier and pass it as the device= value in the testing scripts.(test_microphone_input_ )
    """

    print(sd.query_devices())
