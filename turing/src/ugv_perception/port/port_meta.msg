# ugv_perception/msg/PortMeta.msg — specified in T01, compiled in T07.
# Stream-level hold is the Bool topic, not this message.
std_msgs/Header header    # stamp + frame_id identical to the mask
bool valid
float32 age               # informational only. Freshness = now vs header.stamp
float32 scale             # v1: 1.0
