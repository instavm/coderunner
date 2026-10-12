"""What keeps the assistant from doing harm or leaking the owner's data.

  policy         every action's decision: allow, ask the owner, or deny; approvals and grants
  decision       the System One judge: a decision model asked whether a step commits something
  guard          screening of untrusted text for instructions aimed at the assistant
  egress         the network proxy for the sandbox: allowed hosts, secrets added at the boundary
  secrets_store  secrets kept for the egress proxy, never shown to the sandbox or the model
"""
