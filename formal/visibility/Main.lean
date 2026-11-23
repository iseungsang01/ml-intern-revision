import Visibility

/-- Local-only stdin bridge: query ID, prefix eligibility bit string, expected
    source (`none` or natural). No observations or identifiers leave the machine. -/
def main : IO Unit := do
  let input ← IO.getStdin
  let output ← IO.getStdout
  repeat
    let line ← input.getLine
    if line.isEmpty then break
    let fields := line.trimAscii.toString.splitOn " "
    match fields with
    | [qid, bits, expected] =>
      let flags := bits.toList.toArray.map (· == '1')
      if !(bits.toList.all fun c => c == '0' || c == '1') then
        throw (IO.userError "Invalid eligibility encoding")
      let found := Visibility.latest (fun i => flags[i]?.getD false) flags.size
      let actual := found.map toString |>.getD "none"
      if actual != expected then
        throw (IO.userError s!"Selector mismatch at query {qid}")
      output.putStrLn s!"{qid} {actual}"
    | _ => throw (IO.userError "Invalid bridge query")
