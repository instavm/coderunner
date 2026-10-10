from .base import Channel, IncomingMessage, MessageHandler

__all__ = ["Channel", "IncomingMessage", "MessageHandler", "load_channels"]


def load_channels(config, svc, on_message: MessageHandler) -> dict[str, Channel]:
    """Instantiate every configured channel. Implementations live in sibling modules."""
    from . import imessage, signal, telegram, whatsapp

    registry = {cls.name: cls for cls in (telegram.TelegramChannel, imessage.IMessageChannel,
                                          signal.SignalChannel, whatsapp.WhatsAppChannel)}
    channels: dict[str, Channel] = {}
    for name, settings in (config.get("channels", {}) or {}).items():
        cls = registry.get(name)
        if cls is None or not settings.get("enabled", True):
            continue
        channel = cls(settings, svc, on_message)
        if channel.configured():
            channels[name] = channel
    return channels
