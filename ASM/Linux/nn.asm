; Neural Network asm x64 for Linux

; nasm -f elf64 nn.asm -o nn.o
; gcc nn.o -o nn -no-pie

extern printf
extern exit

section .data
    inputs dq 1.0, 2.0, 3.0
    weights dq 0.5, 0.5, 0.5
    bias dq 1.0
    result dq 0.0
    cutoff dq 0.5
    fmt_out db "Output: %lf", 10, 0
    msg_yes db "Neuron fired!", 10, 0
    msg_no db "Neuron silent.", 10, 0

section .text
global main

main:
    push rbp
    mov rbp, rsp

    xorpd xmm0, xmm0
    xorpd xmm1, xmm1
    
    mov rcx, 3
.loop:
    movsd xmm1, [inputs + rcx*8 - 8]
    mulsd xmm1, [weights + rcx*8 - 8]
    addsd xmm0, xmm1
    loop .loop
    
    addsd xmm0, [bias]
    movsd [result], xmm0
    
    mov rdi, fmt_out
    movq xmm0, [result]
    mov rax, 1
    call printf

    movsd xmm1, [cutoff]
    ucomisd xmm0, xmm1
    jae .fire
    
    mov rdi, msg_no
    xor rax, rax
    call printf
    jmp .done
    
.fire:
    mov rdi, msg_yes
    xor rax, rax
    call printf
    
.done:
    xor rdi, rdi
    call exit

    mov rsp, rbp
    pop rbp
    ret
