require 'cunn';
require('nngraph')
stringx = require('pl.stringx')
require 'io'

-- parameters used for the pretrained model
params = {batch_size=32,
          seq_length=35,
          layers=2,
          decay=1.15,
          rnn_size=1000,
          dropout=0.65,
          init_weight=0.05,
          mom=0.9,
          lr=0.1,
          vocab_size=50,
          max_epoch=4,
          max_max_epoch=50,
          max_grad_norm=5}

-- function to move data to GPU
function transfer_data(x)
  return x:cuda()
end

function g_disable_dropout(node)
  if type(node) == "table" and node.__typename == nil then
    for i = 1, #node do
      node[i]:apply(g_disable_dropout)
    end
    return
  end
  if string.match(node.__typename, "Dropout") then
    node.train = false
  end
end

function g_replace_table(to, from)
  assert(#to == #from)
  for i = 1, #to do
    to[i]:copy(from[i])
  end
end

function reset_state()
  if model ~= nil and model.start_s ~= nil then
    for d = 1, 2 * params.layers do
      model.start_s[d]:zero()
    end
  end
end

function readline()
  local line = io.read("*line")
  if line == nil then error({code="EOF"}) end
  line = stringx.split(line)
  for i = 2,#line do
    -- check to see if the character is in the vocabulary
    if not ptb.vocab_map[line[i]] then error({code="vocab", word = line[i]}) end
  end
  return line
end

-- load pretrained model
model = torch.load('lmodel_4.net')

-- load vocab map
file = torch.DiskFile('vocab_map.asc', 'r')
vocab_map = file:readObject()

function main()
  while true do
    print("OK GO")
    local ok, line = pcall(readline)
    if not ok then
      if line.code == "EOF" then
        break -- end loop
      elseif line.code == "vocab" then
        print("Character not in vocabulary: ", line.word)
      else
        print(line)
        print("Failed, try again")
      end
    else
      -- prepare a few things
      len = line[1]
      reset_state(state_test)
      g_disable_dropout(model.rnns)
      g_replace_table(model.s[0], model.start_s)
      -- tensors to hold words
      local x = transfer_data(torch.zeros(params.batch_size))
      -- doesnt matter what y is
      local y = transfer_data(torch.ones(params.batch_size))
      -- first loop adding the entered words into memory
      for i = 2, #line do 
        -- word that will be used to predict the next
        predictor = line[i]
        local idx = ptb.vocab_map[predictor]
        -- fill all the samples with the same word
        for i=1,params.batch_size do x[i] = idx end
        local s = model.s[i - 1]
        perp_tmp, model.s[1], pred_tmp = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
        -- replace initial state for next iteration with state just generated
        g_replace_table(model.s[0], model.s[1])
        io.write(line[i]..' ') 
      end
      -- generate next word in sequence
      for i = 1,len do
        -- get the index in the vocab map of the word
        idx = ptb.vocab_map[predictor]
        for i=1,params.batch_size do x[i] = idx end
        local s = model.s[i - 1]
        perp_tmp, model.s[1], pred_tmp = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
        -- get the predicted word and print it
        -- _, argmax = pred_tmp[1]:max(1)
        -- io.write(ptb.inverse_vocab_map[argmax[1]]..' ') 
        xx = pred_tmp[1]:clone():float()
        xx = torch.multinomial(torch.exp(xx),1)
        io.write(ptb.inverse_vocab_map[xx[1]]..' ')
        -- replace initial state for next iteration with state just generated
        g_replace_table(model.s[0], model.s[1])
        predictor = ptb.inverse_vocab_map[xx[1]]
      end
      io.write('\n')
    end
  end
end

-- main()