require 'cunn'
ptb = require('data')
stringx = require('pl.stringx')
require 'io'

function transfer_data(x)
  return x:cuda()
end

state_train = {data=transfer_data(ptb.traindataset(1))}

function readline()
  local line = io.read("*line")
  if line == nil then error({code="EOF"}) end
  line = stringx.split(line)
  if tonumber(line[1]) == nil then error({code="init"}) end
  for i = 2,#line do
    -- check to see if the word is in the vocabulary
    if not ptb.vocab_map[line[i]] then error({code="vocab", word = line[i]}) end
  end
  return line
end

while true do
  print("Query: len word1 word2 etc")
  local ok, line = pcall(readline)
  if not ok then
    if line.code == "EOF" then
      break -- end loop
    elseif line.code == "vocab" then
      print("Word not in vocabulary: ", line.word)
    elseif line.code == "init" then
      print("Start with a number")
    else
      print(line)
      print("Failed, try again")
    end
  else
    print("Thanks, I will print now")
    p = 2
    for i = 1, line[1] do 
      io.write(line[p]..' ') 
      p = p+1
    end
    x = #line
    io.write('length of line is '..x)
    io.write('\n')
  end
end